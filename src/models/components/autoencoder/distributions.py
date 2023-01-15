import torch
import numpy as np
from torch import nn


class DiagonalGaussianDistribution(nn.Module):
    def sample(self, mean, logvar, determenistic=False):
        if determenistic:
            std = 0
        else:
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
        x = mean + std * torch.randn_like(mean)
        return x

    def kl(self, mean, logvar):
        logvar = torch.clamp(logvar, -30.0, 20.0)
        var = torch.exp(logvar)
        return 0.5 * torch.sum(torch.pow(mean, 2)
                               + var - 1.0 - logvar,
                               dim=[1, 2, 3]).mean()
