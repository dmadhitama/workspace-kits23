import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    Class representing a network of CNN blocks
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.seq_block(x)
        return x