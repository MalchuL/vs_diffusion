import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold
        self.distance = nn.CosineSimilarity()

    def forward(self, pred, target):
        distance = self.distance(pred, target)
        distance = 1 - distance
        # When we close to target we mask result
        distance[distance < self.threshold] = 0
        loss = distance.mean()
        return loss
