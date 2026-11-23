import torch
import torch.nn as nn
from . import block as B
from .Blocks import BSConvU

class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = BSConvU(in_nc, nf, kernel_size=3)


        self.B1 = B.Modified_RFDB(in_channels=nf)
        self.B2 = B.Modified_RFDB(in_channels=nf)
        self.B3 = B.Modified_RFDB(in_channels=nf)
        self.B4 = B.Modified_RFDB(in_channels=nf)


        self.c = B.sequential(nn.Conv2d(nf * num_modules, nf, kernel_size=1), nn.GELU())
        self.LR_conv = BSConvU(nf, nf, kernel_size=3)

        self.upsampler = B.pixelshuffle_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        return output
