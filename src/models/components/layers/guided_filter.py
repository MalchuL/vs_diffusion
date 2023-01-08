import torch.nn as nn
import torch
import torch.nn.functional as F


# Fdgf from https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf
class GuidedFilter(nn.Module):
    def __init__(self, r=1, input_channels=3, eps=1e-2):
        super().__init__()
        self.r = r
        self.input_channels = input_channels
        self.eps = eps
        self.register_buffer('box_kernel', self.calculate_box_filter(self.r, self.input_channels))
        self.register_buffer('N_box_kernel', self.calculate_box_filter(self.r, 1))

    @staticmethod
    def calculate_box_filter(r, ch):
        weight = 1 / ((2 * r + 1) ** 2)
        box_kernel = weight * torch.ones([ch, 1, 2 * r + 1, 2 * r + 1], dtype=torch.float32)
        return box_kernel

    def box_filter(self, x, channels=None):
        return F.conv2d(x,
                        self.N_box_kernel if channels == 1 else self.box_kernel,
                        bias=None, stride=1, padding='same',
                        groups=self.input_channels if channels is None else channels)

    def forward(self, x, y):
        N,C,H,W = x.shape
        N = self.box_filter(torch.ones(1, 1, H, W).type_as(x), channels=1)
        mean_x = self.box_filter(x) / N
        mean_y = self.box_filter(y) / N

        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        var_x = self.box_filter(x * x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N

        output = mean_A * x + mean_b
        return output

    def get_num_channels(self):
        return 3

if __name__ == '__main__':
    x = torch.rand(2, 3, 64, 64)
    module = GuidedFilter(r=5, input_channels=3)
    print(module(x, x).shape)
