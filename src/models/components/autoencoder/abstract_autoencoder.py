from abc import ABC, abstractmethod

from torch import nn


class AbstractAutoEncoder(nn.Module, ABC):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, z_channels: int, double_z):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_channels = z_channels
        self.double_z = double_z

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, feats):
        return self.decoder(feats)

    def get_encoded_dim(self) -> int:
        return self.z_channels

    def get_double_z(self) -> bool:
        return self.double_z

    def forward(self, x):
        feats = self.forward_encoder(x)
        out = self.forward_decoder(feats)
        return out
