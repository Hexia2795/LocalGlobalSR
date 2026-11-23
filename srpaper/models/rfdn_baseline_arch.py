from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


def conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> nn.Conv2d:
    padding = ((kernel_size - 1) // 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    )


class ESA(nn.Module):


    def __init__(self, n_feats: int) -> None:
        super().__init__()
        f = n_feats // 4
        if f < 1:
            raise ValueError(f"n_feats must be at least 4, got {n_feats}")

        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(
            c3,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        cf = self.conv_f(c1_)
        mask = self.sigmoid(self.conv4(c3 + cf))
        return x * mask


class RFDB(nn.Module):


    def __init__(self, in_channels: int, distillation_rate: float = 0.5) -> None:
        super().__init__()
        if not (0.0 < distillation_rate < 1.0):
            raise ValueError("distillation_rate must be in (0, 1)")

        distilled_channels = max(1, int(in_channels * distillation_rate))
        remaining_channels = in_channels

        self.c1_d = conv_layer(in_channels, distilled_channels, 1)
        self.c1_r = conv_layer(in_channels, remaining_channels, 3)

        self.c2_d = conv_layer(remaining_channels, distilled_channels, 1)
        self.c2_r = conv_layer(remaining_channels, remaining_channels, 3)

        self.c3_d = conv_layer(remaining_channels, distilled_channels, 1)
        self.c3_r = conv_layer(remaining_channels, remaining_channels, 3)

        self.c4 = conv_layer(remaining_channels, distilled_channels, 3)
        self.c5 = conv_layer(distilled_channels * 4, in_channels, 1)

        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.esa = ESA(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.act(self.c1_r(x) + x)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat(
            [distilled_c1, distilled_c2, distilled_c3, r_c4],
            dim=1,
        )
        return self.esa(self.c5(out))


class RFDN(nn.Module):


    def __init__(
        self,
        in_nc: int = 3,
        nf: int = 48,
        num_modules: int = 4,
        out_nc: int = 3,
        upscale: int = 4,
        distillation_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if num_modules < 1:
            raise ValueError("num_modules must be at least 1")
        if upscale not in {2, 3, 4}:
            raise ValueError(f"Supported upscale factors are 2, 3, 4; got {upscale}")

        self.in_nc = in_nc
        self.nf = nf
        self.num_modules = num_modules
        self.out_nc = out_nc
        self.upscale = upscale

        self.fea_conv = conv_layer(in_nc, nf, 3)
        self.blocks = nn.ModuleList(
            [RFDB(nf, distillation_rate=distillation_rate) for _ in range(num_modules)]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(nf * num_modules, nf, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )
        self.lr_conv = conv_layer(nf, nf, 3)
        self.upsampler = nn.Sequential(
            conv_layer(nf, out_nc * (upscale ** 2), 3),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.fea_conv(x)

        block_outputs: List[torch.Tensor] = []
        out = shallow
        for block in self.blocks:
            out = block(out)
            block_outputs.append(out)

        fused = self.fusion(torch.cat(block_outputs, dim=1))
        out_lr = self.lr_conv(fused) + shallow
        return self.upsampler(out_lr)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = RFDN(nf=48, num_modules=4, upscale=4)
    x = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        y = model(x)
    print(model)
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
