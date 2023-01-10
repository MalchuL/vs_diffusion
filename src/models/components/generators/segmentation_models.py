from math import log2, sqrt

import segmentation_models_pytorch as smp
import torch
from torch import nn
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from src.models.components.layers.coordconv import AddCoords

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, affine=True):
        # num_channels = groups * (groups/ DIVIDER)
        groups = 2 ** round((log2(sqrt(num_channels * 2))))
        num_groups = groups
        super().__init__(num_groups, num_channels, affine=affine)
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

def fix_padding(model: nn.Module, padding='reflect'):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            print(module)
            module.padding_mode = padding

class SegmentationModelUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", input_channels=3, output_channels=3):
        super().__init__()
        self.coord_conv = AddCoords()
        _BN = nn.BatchNorm2d
        nn.BatchNorm2d = GroupNorm
        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=input_channels + 1 + 2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=output_channels,  # model output channels (number of classes in your dataset)
        )
        fix_padding(self.model)
        nn.BatchNorm2d = _BN
        print(nn.BatchNorm2d, _BN)

    def forward(self, x, step):
        x = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).type_as(x) * step], dim=1)
        x = self.coord_conv(x)
        return self.model(x)

    @classmethod
    def get_from_params(cls, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=3):
        return SegmentationModelUNet(encoder_name=encoder_name, encoder_weights=encoder_weights, input_channels=in_channels, output_channels=out_channels)
