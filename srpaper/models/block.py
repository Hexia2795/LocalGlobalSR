import torch
import torch.nn as nn
from .Blocks import BSConvU
from .dlgsanet_arch import IDynamicLayerBlock, SparseAttentionLayerBlock

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class BSRB(nn.Module):

    def __init__(self, num_feat):
        super(BSRB, self).__init__()
        self.conv1 = BSConvU(num_feat, num_feat, kernel_size=3, with_ln=False)
        self.act = nn.GELU()
        self.conv2 = BSConvU(num_feat, num_feat, kernel_size=3, with_ln=False)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class Modified_RFDB(nn.Module):

    def __init__(self, in_channels, distillation_rate=0.5):
        super(Modified_RFDB, self).__init__()
        self.dc = int(in_channels * distillation_rate)


        self.d1 = nn.Conv2d(in_channels, self.dc, kernel_size=1)
        self.r1 = BSRB(in_channels)

        self.d2 = nn.Conv2d(in_channels, self.dc, kernel_size=1)
        self.r2 = BSRB(in_channels)

        self.d3 = nn.Conv2d(in_channels, self.dc, kernel_size=1)
        self.r3 = BSRB(in_channels)


        self.d4 = BSConvU(in_channels, self.dc, kernel_size=3)

        self.act = nn.GELU()
        self.c1 = nn.Conv2d(self.dc * 4, in_channels, 1)

        self.local_attn = IDynamicLayerBlock(dim=in_channels, window_size=7, idynamic_num_heads=4)
        self.global_attn = SparseAttentionLayerBlock(dim=in_channels, restormer_num_heads=4, activation='relu')

    def forward(self, x):
        d1 = self.d1(x)
        r1 = self.r1(x)
        d2 = self.d2(r1)
        r2 = self.r2(r1)
        d3 = self.d3(r2)
        r3 = self.r3(r2)
        d4 = self.d4(r3)

        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.act(self.c1(out))


        out = self.local_attn(out)
        out = self.global_attn(out)

        return out + x
