import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.conv_layers import conv3x3 as conv3x3, conv1x1 as conv1x1, conv4x4

NEG_SLOPE = 0.2
LAST_BIAS = False


# Don't change LeakyRelu
class SimpleCartoonGANDiscriminator(nn.Module):
    def __init__(self, use_sigmoid,
                 input_channels=3,
                 layers=[32, 64, 128, 128, 256, 256],
                 strides=[1, 2, 1, 2, 1],
                 mean=[0.0],
                 std=[1.0],
                 groups=1,
                 multiplier=1,
                 eps=1e-5,
                 use_padding=True,
                 apply_instance_norm=False):
        super(SimpleCartoonGANDiscriminator, self).__init__()
        self.eps = eps
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.groups = groups

        def r(x):
            return round(x * multiplier) * groups

        def get_act(inplace=True):
            return nn.LeakyReLU(negative_slope=NEG_SLOPE,
                                inplace=inplace)  # alpha because at -1 gradient lokks like leaky relu

        def get_norm(channels, affine=False):
            return nn.InstanceNorm2d(channels, affine=affine) if apply_instance_norm else nn.BatchNorm2d(channels,
                                                                                                         affine=affine,
                                                                                                         eps=self.eps,
                                                                                                         track_running_stats=False)

        def get_reverse_norm(channels, affine=True):
            return nn.Identity()

        layers = [r(layer) for layer in layers]
        in_channels = [input_channels] + layers[:-1]
        out_channels = layers
        strides = list(strides)
        modules = []
        for i in range(len(strides)):
            use_norm = i > 0 and strides[i] == 1
            conv_fn = conv3x3
            modules.extend(
                [conv_fn(in_channels[i], out_channels[i], stride=strides[i], bias=False, sn=False,
                         use_padding=use_padding, groups=groups),
                 get_norm(out_channels[i]) if use_norm else get_reverse_norm(out_channels[i]),
                 get_act(inplace=use_norm)])

        modules.append(conv3x3(out_channels[-1],
                               out_planes=groups,
                               groups=groups,
                               stride=1,
                               bias=LAST_BIAS,
                               use_padding=use_padding))  # You must have bias as true to good converge
        self.use_sigmoid = use_sigmoid
        self.conv = nn.ModuleList(modules)

    def forward(self, x, apply_sigmoid=True):
        x = (x - self.mean) / self.std
        for i, module in enumerate(self.conv):
            x = module(x)

        if self.use_sigmoid and apply_sigmoid:
            x = torch.sigmoid(x)

        return x
