import torch
import torch.nn as nn

from models.cnnblocks import CNNBlocks

class Encoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding,
                 downhill=4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()

        for _ in range(downhill):
            self.enc_layers += [
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    nn.MaxPool2d(2, 2)
                ]

            in_channels = out_channels
            out_channels *= 2
        # doubling the dept of the last CNN block
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels,
                                         out_channels=out_channels, padding=padding))

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection