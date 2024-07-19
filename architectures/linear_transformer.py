from functools import partial

import torch
from timm.models import register_model
from torch import nn
from timm.models.vision_transformer import Block

from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        feature_map=lambda x: torch.nn.functional.elu(x) + 1,
        eps=1e-6,
    ):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map
        self.eps = eps
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )  # 3 x B x H x N x d_head
        q, k, v = qkv.unbind(0)  # each B x H x N x d_head

        # Apply the feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Compute the KV matrix
        kv = torch.einsum("BHNd,BHNm->BHmd", k, v)  # B x H x d_head x d_head
        kv = self.attn_drop(kv)

        # Compute the normalizer
        z = 1 / (torch.einsum("BHNd,BHd->BHN", q, k.sum(dim=-2)) + self.eps)  # B x H x N

        # Compute final values
        y = torch.einsum(
            "BHNd,BHmd,BHN->BHNm", q, kv, z
        )  # B x H x N x d  # TODO: test with 'transposed' kv (BHmd -> BHdm)
        y = y.transpose(1, 2).reshape(B, N, C)  # B x N x C
        y = self.proj(y)
        y = self.proj_drop(y)

        return y


class LinearAttentionBlock(Block):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, drop=None, **kwargs):
        if drop is not None:
            super().__init__(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, **kwargs)
            proj_drop = drop
        else:
            super().__init__(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, **kwargs)
        self.attn = LinearAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )


class LinearViT(TimmViT):
    def __init__(self, img_size=224, **kwargs):
        super().__init__(img_size=img_size, block_fn=LinearAttentionBlock, **kwargs)


@register_model
def linear_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = LinearViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def linear_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = LinearViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def linear_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = LinearViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def linear_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = LinearViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model
