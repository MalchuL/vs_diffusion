import torch.nn as nn
import torch

from src.models.components.layers.sobel import SobelFilter


# Frcs from https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf
class ColorShift(nn.Module):
    def __init__(self, weight_mode='uniform', is_repeat=True):
        super().__init__()
        self.is_repeat = is_repeat
        assert weight_mode in ['normal', 'uniform']
        self.weight_mode = weight_mode

    @staticmethod
    def random_normal(*size, mean=0.0, stddev=1.0, device=None, dtype=None):
        return torch.randn(*size, dtype=dtype, device=device) * stddev + mean

    @staticmethod
    def random_uniform(*size, minval=0.0, max_val=1.0, device=None, dtype=None):
        return (max_val - minval) * torch.rand(*size, dtype=dtype, device=device) + minval

    def forward(self, image):
        N, C, H, W = image.shape
        dtype = image.dtype
        device = image.device
        r, g, b = torch.chunk(image, chunks=3, dim=1)
        if self.weight_mode == 'normal':
            r_weight = self.random_normal(N, 1, 1, 1, mean=0.299, stddev=0.1, dtype=dtype, device=device)
            g_weight = self.random_normal(N, 1, 1, 1, mean=0.587, stddev=0.1, dtype=dtype, device=device)
            b_weight = self.random_normal(N, 1, 1, 1, mean=0.114, stddev=0.1, dtype=dtype, device=device)
        elif self.weight_mode == 'uniform':
            r_weight = self.random_uniform(N, 1, 1, 1, minval=0.199, max_val=0.399, dtype=dtype, device=device)
            g_weight = self.random_uniform(N, 1, 1, 1, minval=0.487, max_val=0.687, dtype=dtype, device=device)
            b_weight = self.random_uniform(N, 1, 1, 1, minval=0.014, max_val=0.214, dtype=dtype, device=device)
        else:
            raise ValueError("Weight_mode must be in ['normal','uniform']")

        output = (r_weight * r + g * g_weight + b * b_weight) / (r_weight + g_weight + b_weight + 1e-6)

        if self.is_repeat:
            output = output.repeat(1, 3, 1, 1)
        return output

    def get_num_channels(self):
        return 3 if self.is_repeat else 1

class ColorShiftWithSobel(ColorShift):
    def __init__(self, weight_mode='uniform'):
        super(ColorShiftWithSobel, self).__init__(weight_mode=weight_mode, is_repeat=False)
        self.sobel = SobelFilter(use_padding=True, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    def forward(self, image):
        gray_shifted = super(ColorShiftWithSobel, self).forward(image)
        sobel_out = self.sobel(image)
        return torch.cat([gray_shifted, sobel_out], dim=1)

    def get_num_channels(self):
        return super(ColorShiftWithSobel, self).get_num_channels() + self.sobel.get_num_channels()

if __name__ == '__main__':
    x = torch.rand(2, 3, 64, 64)
    module = ColorShift()
    print(module(x).shape)
