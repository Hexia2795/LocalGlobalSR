import torch
from torch import nn
from srpaper.models.dlgsanet_arch import IDynamicDWConv, SparseAttention

class MACCounter:

    def __init__(self):

        self.conv_macs = 0
        self.linear_macs = 0


        self.dynamic_filter_macs = 0


        self.attention_matmul_macs = 0

        self.attention_shapes = []

        self.handles = []
        self.original_sparse_forwards = []

    def conv_hook(
        self,
        module,
        inputs,
        output,
    ):

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Unexpected non-tensor "
                "Conv2d output."
            )

        if output.ndim != 4:
            raise RuntimeError(
                "Unexpected Conv2d "
                "output rank."
            )

        batch = int(
            output.shape[0]
        )

        out_channels = int(
            output.shape[1]
        )

        out_h = int(
            output.shape[2]
        )

        out_w = int(
            output.shape[3]
        )

        kernel_h = int(
            module.kernel_size[0]
        )

        kernel_w = int(
            module.kernel_size[1]
        )

        kernel_mul = (
            module.in_channels
            // module.groups
            * kernel_h
            * kernel_w
        )


        macs = (
            batch
            * out_channels
            * out_h
            * out_w
            * kernel_mul
        )

        self.conv_macs += int(
            macs
        )

    def linear_hook(
        self,
        module,
        inputs,
        output,
    ):

        if not isinstance(
            output,
            torch.Tensor,
        ):
            return

        macs = (
            output.numel()
            * module.in_features
        )

        self.linear_macs += int(
            macs
        )

    def dynamic_filter_hook(
        self,
        module,
        inputs,
        output,
    ):

        x = inputs[0]

        if x.ndim != 4:
            raise RuntimeError(
                "Unexpected "
                "IDynamicDWConv input."
            )

        b, c, h, w = [
            int(v)
            for v in x.shape
        ]

        k = int(
            module.kernel_size
        )


        macs = (
            b
            * c
            * h
            * w
            * k
            * k
        )

        self.dynamic_filter_macs += int(
            macs
        )

    def count_sparse_attention(
        self,
        module,
        qkv,
    ):

        if qkv.ndim != 4:
            raise RuntimeError(
                "Unexpected SparseAttention "
                "_forward input."
            )

        b, c3, h, w = [
            int(v)
            for v in qkv.shape
        ]

        if c3 % 3 != 0:
            raise RuntimeError(
                "QKV channels not "
                "divisible by 3."
            )

        dim = (
            c3 // 3
        )

        heads = int(
            module.num_heads
        )

        if dim % heads != 0:
            raise RuntimeError(
                "Attention dim not "
                "divisible by heads."
            )

        head_dim = (
            dim // heads
        )

        n = (
            h * w
        )


        qk_macs = (
            b
            * heads
            * head_dim
            * head_dim
            * n
        )


        av_macs = (
            b
            * heads
            * head_dim
            * head_dim
            * n
        )

        self.attention_matmul_macs += int(
            qk_macs
            + av_macs
        )

        self.attention_shapes.append(
            (
                b,
                c3,
                h,
                w,
            )
        )

    def attach(
        self,
        model,
    ):

        for module in model.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                handle = (
                    module
                    .register_forward_hook(
                        self.conv_hook
                    )
                )

                self.handles.append(
                    handle
                )

            elif isinstance(
                module,
                nn.Linear,
            ):

                handle = (
                    module
                    .register_forward_hook(
                        self.linear_hook
                    )
                )

                self.handles.append(
                    handle
                )

            if isinstance(
                module,
                IDynamicDWConv,
            ):

                handle = (
                    module
                    .register_forward_hook(
                        self.dynamic_filter_hook
                    )
                )

                self.handles.append(
                    handle
                )

            if isinstance(
                module,
                SparseAttention,
            ):

                original_forward = (
                    module._forward
                )

                self.original_sparse_forwards.append(
                    (
                        module,
                        original_forward,
                    )
                )

                def wrapped_forward(
                    qkv,
                    _module=module,
                    _original=original_forward,
                    _counter=self,
                ):

                    _counter.count_sparse_attention(
                        _module,
                        qkv,
                    )

                    return _original(
                        qkv
                    )

                module._forward = (
                    wrapped_forward
                )

    def detach(
        self,
    ):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()

        for (
            module,
            original_forward,
        ) in self.original_sparse_forwards:

            module._forward = (
                original_forward
            )

        self.original_sparse_forwards.clear()

    @property
    def total_macs(self):

        return (
            self.conv_macs
            + self.linear_macs
            + self.dynamic_filter_macs
            + self.attention_matmul_macs
        )
