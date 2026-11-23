import torch
import torch.nn as nn

from . import block as BaseBlock
from . import block_parallel_unit as ParallelBlock
from .Blocks import BSConvU


class RFDN(nn.Module):
    def __init__(
        self,
        in_nc=3,
        nf=48,
        num_modules=4,
        out_nc=3,
        upscale=4
    ):
        super().__init__()

        if num_modules != 4:
            raise ValueError(
                "单位残差系数并行对照固定使用4个模块，"
                f"但收到num_modules={num_modules}"
            )

        self.fea_conv = BSConvU(
            in_nc,
            nf,
            kernel_size=3
        )

        self.B1 = ParallelBlock.ParallelUnit_RFDB(nf)
        self.B2 = ParallelBlock.ParallelUnit_RFDB(nf)
        self.B3 = ParallelBlock.ParallelUnit_RFDB(nf)
        self.B4 = ParallelBlock.ParallelUnit_RFDB(nf)

        self.c = BaseBlock.sequential(
            nn.Conv2d(
                nf * num_modules,
                nf,
                kernel_size=1
            ),
            nn.GELU()
        )

        self.LR_conv = BSConvU(
            nf,
            nf,
            kernel_size=3
        )

        self.upsampler = BaseBlock.pixelshuffle_block(
            nf,
            out_nc,
            upscale_factor=upscale
        )

    def forward(self, x):
        shallow_feature = self.fea_conv(x)

        out_B1 = self.B1(shallow_feature)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        fused_blocks = self.c(
            torch.cat(
                [out_B1, out_B2, out_B3, out_B4],
                dim=1
            )
        )

        low_resolution_feature = (
            self.LR_conv(fused_blocks)
            + shallow_feature
        )

        return self.upsampler(low_resolution_feature)
