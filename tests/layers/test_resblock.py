import pytest
import torch

from src.models.components.layers.res_block import ResnetBlock


def test_resblock():
    in_channels = 10
    out_channels = 20
    bs = 5
    h, w = 32, 32
    block = ResnetBlock(in_channels, out_channels)
    x = torch.rand(bs, in_channels, h, w)
    out = block(x)
    assert list(out.shape) == [bs, out_channels, h, w]
