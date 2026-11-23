import torch
from torch import nn

class ConditioningIsolationBlock(nn.Module):


    def __init__(
        self,
        block,
        beta,
    ):
        super().__init__()

        self.block = block
        self.beta = float(beta)

    def forward(
        self,
        x,
    ):
        block = self.block

        d1 = block.d1(x)
        r1 = block.r1(x)

        d2 = block.d2(r1)
        r2 = block.r2(r1)

        d3 = block.d3(r2)
        r3 = block.r3(r2)

        d4 = block.d4(r3)

        distilled = torch.cat(
            [
                d1,
                d2,
                d3,
                d4,
            ],
            dim=1,
        )

        base = block.act(
            block.c1(
                distilled
            )
        )


        local_feature = (
            block.local_attn(
                base
            )
        )

        local_delta = (
            local_feature
            - base
        )


        condition = (
            base
            + self.beta
            * local_delta
        )

        conditioned_global = (
            block.global_attn(
                condition
            )
        )

        global_delta = (
            conditioned_global
            - condition
        )


        out = (
            local_feature
            + global_delta
        )

        return out + x
