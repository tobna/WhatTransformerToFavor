from functools import partial

from reformer_pytorch import LSHSelfAttention
from timm.models import register_model
from timm.models.vision_transformer import Block
from torch import nn

from architectures.transformer import Transformer
from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class ReformerAttention(LSHSelfAttention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__(
            dim,
            heads=num_heads,
            causal=False,
            dropout=attn_drop,
            post_attn_dropout=proj_drop,
            n_local_attn_heads=0,
            **kwargs
        )
        if qkv_bias:
            dim_heads = self.dim_head * self.heads
            self.toqk = nn.Linear(dim, dim_heads, bias=qkv_bias)
            v_dims = dim_heads // self.v_head_repeats
            self.tov = nn.Linear(dim, v_dims, bias=qkv_bias)

        assert not qk_norm, "qk_norm not supported"

    def forward(self, x, **kwargs):
        seq_len_in = x.size(1)
        if x.size(1) % (2 * self.bucket_size) != 0:
            x = nn.functional.pad(x, (0, 0, 0, 2 * self.bucket_size - x.size(1) % (2 * self.bucket_size)))
        return super().forward(x, **kwargs)[:, :seq_len_in]


class ReformerBlock(Block):
    def __init__(
        self, dim, num_heads, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0, drop=None, **kwargs
    ):
        if drop is not None:
            proj_drop = drop
        try:
            super().__init__(
                dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, **kwargs
            )
        except TypeError:
            super().__init__(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, drop=proj_drop, **kwargs)

        self.attn = ReformerAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop
        )


class Reformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, block_fn=ReformerBlock, **kwargs)


class ReformerViT(TimmViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, block_fn=ReformerBlock, **kwargs)


@register_model
def reformer_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    model = ReformerViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def reformer_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    model = ReformerViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def reformer_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["B"]
    model = ReformerViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def reformer_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["L"]
    model = ReformerViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def reformer_lra(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    return Reformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})


@register_model
def reformer_lra_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0)
    return Reformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})


@register_model
def reformer_lra_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.0)
    return Reformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})
