import torch
from torch import nn


class StyledInstanceNorm(nn.Module):
    def __init__(self, eps=0):
        super(StyledInstanceNorm, self).__init__()
        self.eps = eps

    def calculate_mean(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)

    def calculate_std(self, x):
        return torch.std(x, dim=(2, 3), keepdim=True)

    def calculate_mean_std(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        return mean, std

    def forward(self, x, mean=None, std=None):
        """

        :param x: has either shape (b, c, x, y) or shape (b, c, x, y, z)
        :return:
        """

        if mean is None and std is None:
            mean, std = self.calculate_mean_std(x)
        elif mean is None:
            mean = self.calculate_mean(x)
        else:
            std = self.calculate_std(x)

        x = (x - mean) / (std + self.eps)
        return x
