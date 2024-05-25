import torch
import torch.nn as nn

from models.cnnblock import CNNBlock

class CNNBlocks(nn.Module):
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a blocl the number of channels doubles
    """
    def __init__(self,
                 n_conv,
                 in_channels,
                 out_channels,
                 padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):

            self.layers.append(CNNBlock(
                in_channels, 
                out_channels, 
                padding=padding
                )
            )
            # after each convolution we set (next) in_channel to (previous) out_channels
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x