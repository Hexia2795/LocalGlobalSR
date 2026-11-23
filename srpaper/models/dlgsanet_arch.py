import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
import numbers
from einops import rearrange


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple([x] * 2)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


class IDynamicDWConv(nn.Module):


    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super(IDynamicDWConv, self).__init__()
        if channels % group_channels != 0:
            raise ValueError(
                f'channels ({channels}) must be divisible by group_channels ({group_channels}).'
            )

        self.kernel_size = int(kernel_size)
        self.channels = int(channels)
        self.group_channels = int(group_channels)
        self.groups = self.channels // self.group_channels

        reduction_ratio = 4
        hidden_channels = self.channels // reduction_ratio
        if hidden_channels <= 0:
            raise ValueError(f'channels is too small: {channels}')

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels, hidden_channels, 1, bias=bias),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=hidden_channels,
                bias=bias,
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                self.kernel_size ** 2 * self.groups,
                kernel_size=1,
                bias=bias,
            )
        )

    def _dynamic_weight(self, x):
        weight = self.conv2(self.conv1(x))
        b, _, h, w = weight.shape
        return weight.view(
            b,
            self.groups,
            self.kernel_size ** 2,
            h,
            w,
        )

    def _forward_vectorized(self, x, weight):

        b, _, h, w = x.shape
        k = self.kernel_size
        num_pixels = h * w


        patches = F.unfold(
            x,
            kernel_size=k,
            padding=k // 2,
            stride=1,
        )
        patches = patches.view(
            b,
            self.groups,
            self.group_channels,
            k ** 2,
            num_pixels,
        )
        weight = weight.view(b, self.groups, k ** 2, num_pixels)


        out = torch.einsum('bgckn,bgkn->bgcn', patches, weight)
        return out.reshape(b, self.channels, h, w)

    def _forward_low_memory(self, x, weight):

        b, _, h, w = x.shape
        k = self.kernel_size
        pad = k // 2

        x_grouped = x.view(
            b,
            self.groups,
            self.group_channels,
            h,
            w,
        )
        x_padded = F.pad(x_grouped, (pad, pad, pad, pad))
        out = torch.zeros_like(x_grouped)

        idx = 0
        for i in range(k):
            for j in range(k):
                x_shift = x_padded[:, :, :, i:i + h, j:j + w]
                w_shift = weight[:, :, idx:idx + 1, :, :]
                out = out + x_shift * w_shift
                idx += 1

        return out.reshape(b, self.channels, h, w)

    def forward(self, x):
        weight = self._dynamic_weight(x)
        if self.training:
            return self._forward_vectorized(x, weight)
        return self._forward_low_memory(x, weight)


class IDynamicDWConvBlock(nn.Module):
    def __init__(self, dim, window_size, dynamic=True, heads=None, bias=True, input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.window_size = window_size
        self.dynamic = dynamic
        self.heads = heads

        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)

        if dynamic:
            self.conv = IDynamicDWConv(dim, kernel_size=window_size, group_channels=heads, bias=bias)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim,
                                  bias=bias)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv1(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=False, tlc_kernel=48, input_resolution=None):
        super(Attention, self).__init__()
        self.tlc_flag = tlc_flag
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        out = (attn @ v)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            out = self.project_out(out)
            return out
        qkv = self.grids(qkv)
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)
        out = self.project_out(out)
        return out


    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        k1, k2 = self.kernel_size
        k1, k2 = min(h, k1), min(w, k2)

        if hasattr(self, '_cached_h_w') and self._cached_h_w == (h, w):
            parts = [x[:, :, i:i + k1, j:j + k2] for i, j in self._cached_ij]
            return torch.cat(parts, dim=0)

        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        cached_ij = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h: i, last_i = h - k1, True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w: j, last_j = w - k2, True
                cached_ij.append((i, j))
                j += step_j
            i += step_i

        self._cached_h_w = (h, w)
        self._cached_ij = cached_ij

        parts = [x[:, :, i:i + k1, j:j + k2] for i, j in cached_ij]
        return torch.cat(parts, dim=0)


    def grids_inverse(self, outs):
        b, c, h, w = self.original_size
        device = outs.device
        k1, k2 = self.kernel_size
        k1, k2 = min(h, k1), min(w, k2)

        if not hasattr(self, '_cached_count_mt') or self._cached_count_mt.shape != (b, 1, h, w) or self._cached_count_mt.device != device:
            count_mt = torch.zeros((b, 1, h, w), device=device)
            for i, j in self._cached_ij:
                count_mt[0, 0, i:i + k1, j:j + k2] += 1.
            self._cached_count_mt = count_mt

        preds = torch.zeros(self.original_size, device=device)
        for cnt, (i, j) in enumerate(self._cached_ij):
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]

        return preds / self._cached_count_mt


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=False, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SparseAttention, self).__init__()
        self.tlc_flag = tlc_flag
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            out = self.project_out(out)
            return out
        qkv = self.grids(qkv)
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)
        out = self.project_out(out)
        return out


    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        k1, k2 = self.kernel_size
        k1, k2 = min(h, k1), min(w, k2)

        if hasattr(self, '_cached_h_w') and self._cached_h_w == (h, w):
            parts = [x[:, :, i:i + k1, j:j + k2] for i, j in self._cached_ij]
            return torch.cat(parts, dim=0)

        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        cached_ij = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h: i, last_i = h - k1, True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w: j, last_j = w - k2, True
                cached_ij.append((i, j))
                j += step_j
            i += step_i

        self._cached_h_w = (h, w)
        self._cached_ij = cached_ij

        parts = [x[:, :, i:i + k1, j:j + k2] for i, j in cached_ij]
        return torch.cat(parts, dim=0)


    def grids_inverse(self, outs):
        b, c, h, w = self.original_size
        device = outs.device
        k1, k2 = self.kernel_size
        k1, k2 = min(h, k1), min(w, k2)

        if not hasattr(self, '_cached_count_mt') or self._cached_count_mt.shape != (b, 1, h, w) or self._cached_count_mt.device != device:
            count_mt = torch.zeros((b, 1, h, w), device=device)
            for i, j in self._cached_ij:
                count_mt[0, 0, i:i + k1, j:j + k2] += 1.
            self._cached_count_mt = count_mt

        preds = torch.zeros(self.original_size, device=device)
        for cnt, (i, j) in enumerate(self._cached_ij):
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]

        return preds / self._cached_count_mt


class IDynamicLayerBlock(nn.Module):
    def __init__(self, dim, window_size=7, idynamic_num_heads=4, idynamic_ffn_type='GDFN',
                 idynamic_ffn_expansion_factor=2., idynamic=True, input_resolution=None):
        super(IDynamicLayerBlock, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.IDynamicDWConv = IDynamicDWConvBlock(dim, window_size=window_size, dynamic=idynamic,
                                                  heads=idynamic_num_heads, input_resolution=input_resolution)
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        if idynamic_ffn_type == 'GDFN':
            self.IDynamic_ffn = FeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=False,
                                            input_resolution=input_resolution)
        elif idynamic_ffn_type == 'BaseFFN':
            self.IDynamic_ffn = BaseFeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=True)

    def forward(self, x):
        x = self.IDynamicDWConv(self.norm1(x)) + x
        x = self.IDynamic_ffn(self.norm2(x)) + x
        return x


class SparseAttentionLayerBlock(nn.Module):

    def __init__(self, dim, restormer_num_heads=4, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2.,
                 tlc_flag=False, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SparseAttentionLayerBlock, self).__init__()
        self.dim = dim
        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.restormer_attn = SparseAttention(dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag,
                                              tlc_kernel=tlc_kernel, activation=activation,
                                              input_resolution=input_resolution)
        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')
        if restormer_ffn_type == 'GDFN':
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False,
                                             input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x


class DLGSANet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
