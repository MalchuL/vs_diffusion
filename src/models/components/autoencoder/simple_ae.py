import numpy as np
import torch.nn as nn

from src.models.components.autoencoder.abstract_autoencoder import AbstractAutoEncoder
from src.models.components.layers.conv_layers import conv3x3
from src.models.components.layers.res_block import ResnetBlock, Downsample, Upsample


class SimpleAutoEncoder(AbstractAutoEncoder):
    def __init__(self, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True,
                 z_channels=4, double_z=True, norm_layer=nn.BatchNorm2d):
        encoder = Encoder(in_channels=in_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                          dropout=dropout,
                          resamp_with_conv=resamp_with_conv, z_channels=z_channels, double_z=double_z,
                          norm_layer=norm_layer)
        decoder = Decoder(out_ch=in_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                          dropout=dropout, resamp_with_conv=resamp_with_conv, z_channels=z_channels, give_pre_end=False,
                          norm_layer=norm_layer)
        self.double_z = double_z
        super().__init__(encoder=encoder, decoder=decoder, z_channels=z_channels)

    def forward(self, x):
        feats = self.forward_encoder(x)
        if self.double_z:
            feats = feats[:, :self.get_encoded_dim(), :, :]

        out = self.forward_decoder(feats)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True,
                 z_channels=4, double_z=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = conv3x3(in_channels, self.ch)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        blocks = []
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                blocks.append(ResnetBlock(in_channels=block_in,
                                          out_channels=block_out,
                                          temb_channels=self.temb_ch,
                                          dropout=dropout,
                                          norm_layer=norm_layer))
                block_in = block_out
            if i_level != self.num_resolutions - 1:
                blocks.append(Downsample(block_in, resamp_with_conv))
        self.blocks = nn.Sequential(*blocks)

        # middle
        self.block_1 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout,
                                   norm_layer=norm_layer)
        self.block_2 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout,
                                   norm_layer=norm_layer)

        # end
        self.norm_out = norm_layer(block_in)
        self.conv_out = conv3x3(block_in,
                                2 * z_channels if double_z else z_channels,
                                stride=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv_in(x)
        out = self.blocks(x)

        # middle
        h = out
        h = self.block_1(h)
        h = self.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_ch, ch, ch_mult=(1, 2, 4, 4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True,
                 z_channels=4, give_pre_end=False, norm_layer=nn.BatchNorm2d
                 ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = conv3x3(z_channels,
                               block_in,
                               stride=1)

        # middle
        self.block_1 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   dropout=dropout,
                                   norm_layer=norm_layer)
        self.block_2 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   dropout=dropout,
                                   norm_layer=norm_layer)

        # upsampling
        blocks = []
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in,
                                          out_channels=block_out,
                                          dropout=dropout,
                                          norm_layer=norm_layer))
                block_in = block_out

            if i_level != 0:
                blocks.append(Upsample(block_in, resamp_with_conv))
        self.blocks = nn.Sequential(*blocks)

        # end
        self.norm_out = norm_layer(block_in)
        self.conv_out = conv3x3(block_in,
                                out_ch,
                                stride=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]

        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.block_1(h, temb)
        h = self.block_2(h, temb)

        # upsampling
        h = self.blocks(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        return h
