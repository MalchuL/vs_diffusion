import math
from typing import List, Optional

import torch
from torch import nn

from src.models.components.layers.conv_layers import conv3x3
from src.models.components.layers.coordconv import AddCoords
from src.models.components.utils.norm_layers import get_norm_layer
from src.models.components.utils.pool_layers import get_pool_layer

PADDING_MODE = 'reflect'
BIAS = False

NEG_SLOPE = 0.1
MAIN_ACTIVATION = lambda: nn.LeakyReLU(negative_slope=NEG_SLOPE, inplace=True)
CONCAT_CHANNELS = True
CONV_UPSAMPLE = True


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=nn.BatchNorm2d,
                 bias=False):
        super().__init__()
        conv = nn.Conv2d(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=kernel_size // 2,
                         padding_mode=PADDING_MODE,
                         bias=bias,
                         groups=groups)
        norm = norm_layer(out_channels)
        act = MAIN_ACTIVATION()

        self.model = nn.Sequential(conv, norm, act)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = MAIN_ACTIVATION()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x, shortcut=None, children: Optional[List[torch.Tensor]] = None):
        if shortcut is None:
            shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut, norm_layer=nn.BatchNorm2d):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = norm_layer(out_channels)
        self.relu = MAIN_ACTIVATION()
        self.shortcut = shortcut

    def forward(self, x_children: List[torch.Tensor]):
        x = self.conv(torch.cat(x_children, 1))
        x = self.bn(x)
        if self.shortcut:
            x += x_children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
            self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1,
            root_shortcut=False, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = pool_layer(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, norm_layer=norm_layer)
            self.tree2 = block(out_channels, out_channels, 1, norm_layer=norm_layer)
            if in_channels != out_channels:
                # NOTE the official impl/weights have  project layers in levels > 1 case that are never
                # used, I've moved the project layer here to avoid wasted params but old checkpoints will
                # need strict=False while loading.
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    norm_layer(out_channels))
            self.root = Root(root_dim, out_channels, root_kernel_size, root_shortcut, norm_layer=norm_layer)
        else:
            self.tree1 = Tree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size,
                root_shortcut=root_shortcut, norm_layer=norm_layer, pool_layer=nn.MaxPool2d)
            self.tree2 = Tree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size, root_shortcut=root_shortcut, norm_layer=norm_layer,
                pool_layer=nn.MaxPool2d)
            self.root = None
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, shortcut: Optional[torch.Tensor] = None, children: Optional[List[torch.Tensor]] = None):
        if children is None:
            children = []
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:  # levels == 1
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x


class IDAUp(nn.Module):

    def __init__(self, channels, norm_layer=nn.BatchNorm2d):
        super(IDAUp, self).__init__()
        self.channels = list(reversed(channels))
        for i in range(len(self.channels) - 1):
            c = self.channels[i]
            o = self.channels[i + 1]
            f = 2
            proj = ConvBNRelu(c, o, norm_layer=norm_layer)
            if CONCAT_CHANNELS:
                node = ConvBNRelu(2 * o, o, norm_layer=norm_layer)
            else:
                node = ConvBNRelu(o, o, norm_layer=norm_layer)

            if CONV_UPSAMPLE:
                up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                        padding=f // 2, output_padding=0,
                                        groups=o, bias=False)
            else:
                up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers):
        for i in range(len(self.channels) - 1):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            upsampled_layer = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i))
            if CONCAT_CHANNELS:
                layers[i + 1] = node(torch.cat([upsampled_layer, layers[i + 1]], dim=1))
            else:
                layers[i + 1] = node(upsampled_layer + layers[i + 1])


class DLAUp(nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d):
        super(DLAUp, self).__init__()
        channels = list(channels)
        self.channels = channels
        for i in range(1, len(self.channels)):
            setattr(self,
                    'ida_{}'.format(i),
                    IDAUp(channels[:i + 1], norm_layer=norm_layer))

    def forward(self, layers):
        out = [layers[0]]  # start with 32
        for i in range(1, len(self.channels)):
            out.insert(0, layers[i])
            ida = getattr(self, 'ida_{}'.format(i))
            ida(out)

        return out[-1]


class DLAHead(nn.Module):
    def __init__(self, upsample_count, in_channels, head_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.upsample_count = upsample_count

        UPSAMPLE_SIZE = 2
        modules = []
        for i in range(self.upsample_count):
            modules.extend([nn.ConvTranspose2d(in_channels, in_channels, UPSAMPLE_SIZE * 2, stride=UPSAMPLE_SIZE,
                                               padding=UPSAMPLE_SIZE // 2, output_padding=0,
                                               groups=1, bias=False),
                            nn.LeakyReLU(negative_slope=0.2)])
        cur_channel = in_channels
        for channel in head_channels:
            modules.append(ConvBNRelu(cur_channel, channel, norm_layer=norm_layer))
            cur_channel = channel

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class VSGenerator(nn.Module):
    def __init__(self,
                 input_channels=3,
                 output_channels=3,
                 apply_tanh=False,
                 channels=[16, 32, 64, 128, 256, 512],
                 head_channels=[32, 16],
                 levels=[1, 1, 1, 1, 1, 1],
                 ida_channels=None,
                 norm_name='batch',
                 pool_name='avg',
                 padding_mode=PADDING_MODE):
        super().__init__()
        global PADDING_MODE
        PADDING_MODE = padding_mode
        assert len(channels) == len(levels) == 6
        self.channels = channels
        self.layers = levels
        self.apply_tanh = apply_tanh
        norm_layer = get_norm_layer(norm_name)
        pool_layer = get_pool_layer(pool_name)

        self.coord_conv = AddCoords()

        self.base_layer = ConvBNRelu(input_channels + 1 + 2, channels[0], kernel_size=7, norm_layer=norm_layer)

        donwsample_size = 2

        self.last_level = round(math.log2(donwsample_size))

        self.ida_channels = channels[self.last_level] if ida_channels is None else ida_channels
        if ida_channels is None:
            self.last_level_mapper = nn.Identity()
        else:
            self.last_level_mapper = ConvBNRelu(channels[self.last_level], self.ida_channels, norm_layer=norm_layer, bias=BIAS)

        self.module_levels = nn.ModuleList()

        level0 = self._make_conv_level(channels[0] + 1 + 2, channels[0], levels[0], norm_layer=norm_layer)
        level1 = self._make_conv_level(channels[0] + 1 + 2, channels[1], levels[1], stride=2, norm_layer=norm_layer)

        block = ResidualBlock

        level2 = Tree(levels[2], block, channels[1] + 1 + 2, channels[2], 2, level_root=False, norm_layer=norm_layer,
                      pool_layer=pool_layer)
        level3 = Tree(levels[3], block, channels[2] + 1 + 2, channels[3], 2, level_root=True, norm_layer=norm_layer,
                      pool_layer=pool_layer)
        level4 = Tree(levels[4], block, channels[3] + 1 + 2, channels[4], 2, level_root=True, norm_layer=norm_layer,
                      pool_layer=pool_layer)
        level5 = Tree(levels[5], block, channels[4] + 1 + 2, channels[5], 2, level_root=True, norm_layer=norm_layer,
                      pool_layer=pool_layer)

        self.module_levels.extend([level0, level1, level2, level3, level4, level5])

        self.dlaup = DLAUp([self.ida_channels] + self.channels[self.last_level + 1:],
                           norm_layer=norm_layer)

        self.head = DLAHead(self.last_level,
                            self.ida_channels,
                            head_channels=head_channels,
                            norm_layer=norm_layer)
        self.last_layer = nn.Conv2d(head_channels[-1],
                                    output_channels,
                                    kernel_size=3,
                                    padding=1,
                                    padding_mode=PADDING_MODE)

        if self.apply_tanh:
            self.act = nn.Tanh()
        else:
            self.act = lambda x: x

    @staticmethod
    def _make_conv_level(inplanes, planes, repeats, stride=1, norm_layer=nn.BatchNorm2d):
        modules = []
        for i in range(repeats):
            modules.append(ConvBNRelu(inplanes, planes, stride=stride, norm_layer=norm_layer))
            stride = 1
            inplanes = planes
        return nn.Sequential(*modules)

    def forward_features(self, x, step):
        out_feats = []
        x = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).type_as(x) * step], dim=1)
        x = self.coord_conv(x)
        out = self.base_layer(x)
        for i, module in enumerate(self.module_levels):
            out = torch.cat([out, torch.ones(out.shape[0], 1, out.shape[2], out.shape[3]).type_as(x) * step], dim=1)
            out = self.coord_conv(out)
            out = module(out)
            if i >= self.last_level:
                out_feats.append(out)

        out_feats[0] = self.last_level_mapper(out_feats[0])
        return out_feats

    def forward_head(self, out_feats):
        out = self.dlaup(out_feats)
        out = self.head(out)
        out = self.last_layer(out)
        out = self.act(out)
        return out
    def forward(self, x, step):
        out_feats = self.forward_features(x, step)
        out = self.forward_head(out_feats.copy())
        return out


if __name__ == '__main__':
    model = VSGenerator(norm_name='group', ida_channels=64)
    print(model)
    x = torch.zeros(2, 3, 256, 256)
    out = model(x)
