import torch.nn as nn


PADDING_MODE = 'reflect'
BIAS = False


def conv3x3(in_planes, out_planes, stride=1, bias=BIAS, use_padding=True, groups=1):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1 if use_padding else 0,
                     bias=bias,
                     padding_mode=PADDING_MODE,
                     groups=groups)
    return conv


def conv1x1(in_planes, out_planes, bias=BIAS, groups=1):
    "1x1 convolution with padding"
    conv = nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     groups=groups,
                     bias=bias)
    return conv
