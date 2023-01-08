import torch
from torch import nn

from src.models.components.layers.color_shift import ColorShift, ColorShiftWithSobel
from src.models.components.layers.guided_filter import GuidedFilter


# Style representation module https://arxiv.org/pdf/2207.02426.pdf
class StyleRepresentation(nn.Module):
    def __init__(self, layers=(), *args, weight_mode='uniform', r=5, eps=2e-1):
        super().__init__()
        self.layer_names = layers
        self.layers = nn.ModuleDict({'color_shift': ColorShift(weight_mode=weight_mode),
                                     'color_shift_with_sobel': ColorShiftWithSobel(weight_mode=weight_mode),
                                     'surface': GuidedFilter(r=r, eps=eps),
                                     'identity': nn.Identity()})

    def forward(self, x):
        layers = []
        for layer_name in self.layer_names:
            if layer_name == 'surface':
                out_layer = self.layers[layer_name](x, x)
            else:
                out_layer = self.layers[layer_name](x)
            layers.append(out_layer)
        out = torch.cat(layers, dim=1)
        return out

    def get_num_channels(self):  # Estimated number of channels
        return len(self.layer_names) * 3
