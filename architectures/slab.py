# Taken from https://github.com/xinghaochen/SLAB/blob/main/classification/models/slab_deit.py with slight changes
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block
from einops import rearrange
from architectures.vit import TimmViT
from functools import partial
from timm.models import register_model
from resizing_interface import vit_sizes


class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2).contiguous()
        return x


class LinearNorm(nn.Module):
    def __init__(self, dim, norm1, norm2, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer("warm", torch.tensor(warm))
        self.register_buffer("iter", torch.tensor(step))
        self.register_buffer("total_step", torch.tensor(step))
        self.r0 = r0
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            x = self.norm2(x)
        return x


class SlabAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, token_width, qkv_bias=False, attn_drop=0.1, proj_drop=0.1, focusing_factor=3):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.token_width = token_width

        self.dwc = nn.Conv2d(
            in_channels=head_dim, out_channels=head_dim, kernel_size=5, groups=head_dim, padding=5 // 2
        )
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, token_width * token_width, dim)))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) # B H N C/H
        k = k + self.positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x).contiguous()
        return x


class SlabBlock(Block):
    def __init__(
        self, dim, num_heads, token_width, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, focusing_factor=3, **kwargs
    ):
        super().__init__(dim, num_heads, **kwargs)
        self.attn = SlabAttention(
            dim,
            token_width=token_width,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            focusing_factor=focusing_factor,
        )


class SlabViT(TimmViT):
    def __init__(self, img_size, focusing_factor=3, patch_size=16, **kwargs):
        token_width = img_size // patch_size
        super().__init__(
            img_size,
            patch_size=patch_size,
            block_fn=partial(SlabBlock, focusing_factor=focusing_factor, token_width=token_width),
            global_pool="avg",
            class_token=False,
            **kwargs,
        )

        self.token_width = token_width

    def _set_input_strand(self, res=None, patch_size=None):
        if res == self.img_size and patch_size == self.patch_size:
            return
        super()._set_input_strand(res=res, patch_size=patch_size)

        # resize positional encoding in attention blocks
        token_width = int(self.img_size // self.patch_size)
        for block in self.blocks:
            old_pos_enc = block.attn.positional_encoding.permute(0, 2, 1).reshape(
                1, block.attn.dim, block.attn.token_width, block.attn.token_width
            )  # 1 x dim x H x W
            pos_enc = (
                F.interpolate(old_pos_enc, size=(token_width, token_width), mode="bicubic")
                .reshape(1, block.attn.dim, token_width**2)
                .permute(0, 2, 1)
            )
            block.attn.positional_encoding = nn.Parameter(pos_enc.contiguous())
            block.attn.token_width = token_width


@register_model
def slab_tiny_patch16(img_size=224, pretrained=False, **kwargs):
    assert img_size == 224, f"Slab DeiT only works with img_size=224; Not with {img_size}."
    ln = partial(nn.LayerNorm, eps=1e-6)
    linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)
    sizes = vit_sizes["Ti"]
    model = SlabViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        mlp_ratio=4,
        norm_layer=linearnorm,
        fc_norm=False,
        **{**sizes, **kwargs},
    )
    return model


@register_model
def slab_small_patch16(img_size=224, pretrained=False, **kwargs):
    assert img_size == 224, f"Slab DeiT only works with img_size=224; Not with {img_size}."
    ln = partial(nn.LayerNorm, eps=1e-6)
    linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)
    sizes = vit_sizes["S"]
    model = SlabViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        mlp_ratio=4,
        norm_layer=linearnorm,
        fc_norm=False,
        **{**sizes, **kwargs},
    )
    return model
