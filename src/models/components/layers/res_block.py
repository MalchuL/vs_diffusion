from functools import partial

import torch.nn as nn
import torch

from src.models.components.layers.conv_layers import conv3x3, conv1x1, PADDING_MODE


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = conv3x3(in_channels,
                                in_channels,
                                stride=1,
                                )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv3x3(in_channels,
                                        in_channels,
                                        stride=2,
                                        use_padding=False)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode=PADDING_MODE, value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, norm_layer=nn.BatchNorm2d, temb_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = norm_layer(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        if temb_channels is not None and temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)

        self.norm2 = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.activation = nn.SiLU(inplace=True)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv3x3(in_channels, out_channels)
            else:
                self.nin_shortcut = conv1x1(in_channels, out_channels)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.activation(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
