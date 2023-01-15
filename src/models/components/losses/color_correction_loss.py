from torch import nn


class ColorCorrectionLoss(nn.Module):
    def __init__(self):
        """
        Loss which restricts colors if they low than input, else no changes
        """
        super().__init__()
        self.loss = nn.ReLU()

    def forward(self, fake, target):
        mean_fake = fake.mean((2, 3))
        mean_target = target.mean((2, 3))
        loss = self.loss(mean_target - mean_fake).mean()
        return loss


class SizedColorCorrectionLoss(nn.Module):
    def __init__(self, target_size):
        """
        Loss which restricts colors if they low than input, else no changes
        """
        super().__init__()
        self.target_size = target_size
        self.resizer = nn.AdaptiveAvgPool2d(self.target_size)
        self.loss = nn.ReLU()

    def forward(self, fake, target):
        mean_fake = self.resizer(fake)
        mean_target = self.resizer(target)
        loss = self.loss(mean_target - mean_fake).mean()
        return loss