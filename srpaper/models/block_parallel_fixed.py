import torch
import torch.nn as nn

from .block import BSRB
from .Blocks import BSConvU
from .dlgsanet_arch import IDynamicLayerBlock, SparseAttentionLayerBlock


class ParallelFixed_RFDB(nn.Module):


    def __init__(self, in_channels, distillation_rate=0.5):
        super().__init__()

        self.dc = int(in_channels * distillation_rate)

        self.d1 = nn.Conv2d(
            in_channels,
            self.dc,
            kernel_size=1
        )
        self.r1 = BSRB(in_channels)

        self.d2 = nn.Conv2d(
            in_channels,
            self.dc,
            kernel_size=1
        )
        self.r2 = BSRB(in_channels)

        self.d3 = nn.Conv2d(
            in_channels,
            self.dc,
            kernel_size=1
        )
        self.r3 = BSRB(in_channels)

        self.d4 = BSConvU(
            in_channels,
            self.dc,
            kernel_size=3
        )

        self.act = nn.GELU()

        self.c1 = nn.Conv2d(
            self.dc * 4,
            in_channels,
            kernel_size=1
        )

        self.local_attn = IDynamicLayerBlock(
            dim=in_channels,
            window_size=7,
            idynamic_num_heads=4
        )

        self.global_attn = SparseAttentionLayerBlock(
            dim=in_channels,
            restormer_num_heads=4,
            activation='relu'
        )

    def forward(self, x):
        d1 = self.d1(x)
        r1 = self.r1(x)

        d2 = self.d2(r1)
        r2 = self.r2(r1)

        d3 = self.d3(r2)
        r3 = self.r3(r2)

        d4 = self.d4(r3)

        distilled = torch.cat(
            [d1, d2, d3, d4],
            dim=1
        )

        base = self.act(self.c1(distilled))


        local_feature = self.local_attn(base)
        global_feature = self.global_attn(base)


        local_delta = local_feature - base
        global_delta = global_feature - base

        fused = (
            base
            + 0.5 * local_delta
            + 0.5 * global_delta
        )

        return fused + x
