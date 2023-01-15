import torch.nn as nn


class IdentityLoss(nn.Module):

    def forward(self, *args, **kwargs):
        return 0.0
