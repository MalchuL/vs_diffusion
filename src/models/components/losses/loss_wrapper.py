import torch.nn as nn


class LossWrapper(nn.Module):
    def __init__(self, loss, weight=1):
        super().__init__()
        if weight <= 0:
            self.loss = None
        else:
            self.loss = loss
        self.weight = weight

    def forward(self, *args, **kwargs):
        if self.weight > 0:
            loss = self.loss(*args, **kwargs)
            if isinstance(loss, tuple):
                loss, *out_values = loss
                return loss * self.weight, *out_values
            else:
                return self.loss(*args, **kwargs) * self.weight
        else:
            return 0

    def denorm_loss(self, loss_value):
        if self.weight > 0:
            return loss_value / self.weight
        else:
            return 0