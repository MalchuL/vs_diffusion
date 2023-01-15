import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, criterion=nn.BCELoss(), is_logit=True):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.is_logit = is_logit
        self.base_loss = criterion

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, pred, target_is_real):
        return self.base_loss(pred, self.get_target_tensor(pred, target_is_real))


