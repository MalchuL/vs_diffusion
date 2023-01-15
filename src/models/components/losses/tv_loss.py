import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).mean()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).mean()
        return (h_tv + w_tv)

