# Taken from https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py with slight changes
from torch import nn
from timm.models.vision_transformer import Block
from architectures.vit import TimmViT
from resizing_interface import vit_sizes
from timm.models import register_model


class HydraAttention(nn.Module):
    def __init__(self, dim, output_layer="linear", drop=0.0, proj_drop=None, qkv_bias=True):
        super(HydraAttention, self).__init__()
        dropout = proj_drop if proj_drop is not None else drop
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out = nn.Linear(dim, dim) if output_layer == "linear" else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, D)"""
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2)  # dropout in seq dimension
        out = kvw.sum(dim=-2, keepdim=True) * q
        return self.out(out)


class HydraBlock(Block):
    def __init__(self, dim, *args, proj_drop=0.0, drop=None, qkv_bias=False, **kwargs):
        if drop is not None:
            super(HydraBlock, self).__init__(dim, *args, drop=drop, **kwargs)
        else:
            super(HydraBlock, self).__init__(dim, *args, proj_drop=proj_drop, **kwargs)
        self.attn = HydraAttention(dim, proj_drop=proj_drop, drop=drop, qkv_bias=qkv_bias)


class HydraViT(TimmViT):
    def __init__(self, *args, **kwargs):
        super(HydraViT, self).__init__(*args, block_fn=HydraBlock, **kwargs)


@register_model
def hydra_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    size = vit_sizes["Ti"]
    model = HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})
    return model


@register_model
def hydra_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    size = vit_sizes["S"]
    model = HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})
    return model


@register_model
def hydra_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    size = vit_sizes["B"]
    model = HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})
    return model


@register_model
def hydra_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    size = vit_sizes["L"]
    model = HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})
    return model
