import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

class UNET(nn.Module):
    def __init__(self,
                 in_channels,
                 first_out_channels,
                 exit_channels,
                 downhill,
                 padding=0
                 ):
        super(UNET, self).__init__()
        self.encoder = Encoder(
            in_channels, 
            first_out_channels, 
            padding=padding, 
            downhill=downhill
        )
        self.decoder = Decoder(
            first_out_channels*(2**downhill), 
            first_out_channels*(2**(downhill-1)),
            exit_channels, 
            padding=padding, 
            uphill=downhill
        )

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        return out