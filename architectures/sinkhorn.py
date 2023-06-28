from functools import partial
from itertools import chain, combinations
from math import prod

import torch
from sinkhorn_transformer import sinkhorn_transformer
from timm.models import register_model
from torch import nn
import torch.nn.functional as F

from architectures.cait import CaiT
from architectures.vit import TimmViT
from resizing_interface import vit_sizes
from utils import prime_factors


def amp_diff_topk(x, k, temperature=1.):
    # This is a slightly modified version of code from https://github.com/lucidrains/sinkhorn-transformer
    # The original code is licensed under the MIT license (see licenses/MIT.txt) from Phil Wang, 2020.
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x, dtype=values.dtype).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)


sinkhorn_transformer.differentiable_topk = amp_diff_topk


class SinkhornAttn(sinkhorn_transformer.SinkhornSelfAttention):
    def __init__(self, dim, num_heads, bucket_size, qkv_bias, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(dim, bucket_size, -1, use_simple_sort_net=False, attn_dropout=attn_drop, dropout=proj_drop,
                         heads=num_heads, **kwargs)
        if qkv_bias:
            self.to_q = nn.Linear(dim, self.dim_head * self.heads, bias=qkv_bias)
            self.to_kv = nn.Linear(dim, 2 * self.dim_head * self.heads, bias=qkv_bias)


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def _largest_div_leq(n, div_bound):
    factors = prime_factors(n)
    for p in set(factors):
        n = 0
        while p**(n + 1) < div_bound:
            n += 1
        while factors.count(p) > n:
            factors.remove(p)
    power_set = set(_powerset(factors))
    max_prod = 1
    for subset in power_set:
        cur_prod = prod(subset)
        if div_bound >= cur_prod > max_prod:
            max_prod = cur_prod
    return max_prod


class SinkhornViT(CaiT):
    def __init__(self, img_size=224, patch_size=16, bucket_size_leq=32, sinkhorn_iter=8, temperature=0.75, **kwargs):
        assert img_size % patch_size == 0, f"img_size has to be divisible by patch_size"
        bucket_size = _largest_div_leq((img_size // patch_size)**2, bucket_size_leq)
        print(f"Using bucket size {bucket_size}")
        super().__init__(attn_block=partial(SinkhornAttn, bucket_size=bucket_size, sinkhorn_iter=sinkhorn_iter,
                                            temperature=temperature), img_size=img_size, patch_size=patch_size,
                         **kwargs)
        self.patch_size = patch_size
        self.bucket_size_leq = bucket_size_leq
        self.bucket_size = bucket_size

    def set_image_res(self, res):
        super().set_image_res(res)
        new_bucket_size = _largest_div_leq((res // self.patch_size)**2, self.bucket_size_leq)
        for block_layer in self.blocks:
            block_layer.attn.bucket_size = new_bucket_size
            block_layer.attn.kv_bucket_size = new_bucket_size
            block_layer.attn.sinkhorn_attention.bucket_size = new_bucket_size
            block_layer.attn.sinkhorn_attention.kv_bucket_size = new_bucket_size
            block_layer.attn.sinkhorn_attention.sort_net.bucket_size = new_bucket_size
            block_layer.attn.sinkhorn_attention.sort_net.kv_bucket_size = new_bucket_size

            if block_layer.attn.n_local_attn_heads > 0:
                raise NotImplementedError(f"No bucket resizing for local attention implemented.")
        print(f"Changing bucket size {self.bucket_size} |-> {new_bucket_size}")
        self.bucket_size = new_bucket_size


@register_model
def sinkhorn_cait_tiny_bmax32_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=32, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_tiny_bmax64_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=64, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_small_bmax32_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=32, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_small_bmax64_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=64, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_base_bmax32_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=32, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_base_bmax64_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=64, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_large_bmax32_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=32, **sizes, **kwargs)
    return model

@register_model
def sinkhorn_cait_large_bmax64_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = SinkhornViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        bucket_size_leq=64, **sizes, **kwargs)
    return model
