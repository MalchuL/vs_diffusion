import pytest
import torch

from src.models.components.autoencoder.simple_ae import Encoder, Decoder, SimpleAutoEncoder
from src.models.components.layers.res_block import ResnetBlock


def test_encoder():
    in_channels = 10
    out_channels = 16
    bs = 5
    h, w = 256, 256
    block = Encoder(in_channels, out_channels)
    with torch.inference_mode():
        x = torch.rand(bs, in_channels, h, w)
        out = block(x)
        print(out.shape)

def test_decoder():
    in_channels = 4
    out_channels = 3
    bs = 5
    h, w = 32, 32
    block = Decoder(in_channels, out_channels)
    with torch.inference_mode():
        x = torch.rand(bs, in_channels, h, w)
        out = block(x)
        print(out.shape)

def test_ar():
    in_channels = 3
    bs = 5
    h, w = 256, 256
    block = SimpleAutoEncoder(input_channels=in_channels, ch=16)
    with torch.inference_mode():
        x = torch.rand(bs, in_channels, h, w)
        out = block(x)
        print(out.shape)

