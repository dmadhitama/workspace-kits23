import torch.nn as nn
import torch
from torch import Tensor

class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.normalization = nn.Softmax(dim=1)

    # def forward(self, inputs, targets, smooth=1):
    #     inputs = self.normalization(inputs)

    #     targets = targets[:, 1:2, ...]
    #     inputs = torch.where(inputs[:, 1:2, ...] > 0.5, 1.0, 0.0)

    #     inputs = inputs.reshape(-1)
    #     targets = targets.reshape(-1)

    #     intersection = (inputs * targets).sum()
    #     dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    #     return dice
    
    def forward(
        self,
        input: Tensor, 
        target: Tensor, 
        reduce_batch_first: bool = False, 
        epsilon: float = 1e-6
    ):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()