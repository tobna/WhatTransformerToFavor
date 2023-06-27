# based on https://github.com/10-zin/Synthesizer/blob/master/synth/synthesizer/modules.py
import math
from functools import partial

import torch
from timm.models import register_model
from torch import nn, einsum
from torch.nn import functional as F
from timm.models.vision_transformer import Block

from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class FactorizedDenseAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, f, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert max_seq_len % f == 0, f"seq_len ({max_seq_len}) has to be divisible by f ({f})."
        self.f = f
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.dim = dim
        self.f_a = nn.Linear(dim, num_heads * f, bias=qkv_bias)
        self.f_b = nn.Linear(dim, num_heads * max_seq_len // f, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        a = self.f_a(x).reshape(B, N, self.num_heads, self.f).transpose(1, 2)  # B x H x N x f
        b = self.f_b(x).reshape(B, N, self.num_heads, self.max_seq_len // self.f).transpose(1, 2)  # B x H x N x ms/f
        dense_attn = torch.repeat_interleave(a, self.max_seq_len // self.f, -1) * torch.repeat_interleave(b, self.f, -1)  # B x H x N x ms
        dense_attn = dense_attn[:, :, :, :N]  # B x H x N x N
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # B x H x N x d

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.attn_drop(F.softmax(dense_attn, dim=-1))
        output = dense_attn @ v  # B x H x N x d
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class FactorizedDenseBlock(Block):
    def __init__(self, dim, num_heads, max_seq_len, f, qkv_bias=False, proj_drop=0., attn_drop=0., **kwargs):
        super().__init__(dim, num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop, **kwargs)
        self.attn = FactorizedDenseAttention(dim, num_heads, max_seq_len, f, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                             proj_drop=proj_drop)


class FDSynthesizer(TimmViT):
    def __init__(self, img_size=224, patch_size=16, f=None, class_token=True, qkv_bias=False, **kwargs):
        seq_len = (img_size / patch_size) ** 2 + (1 if class_token else 0)
        self.keep_f = f is not None
        if f is None:
            f = round(math.sqrt(seq_len))
        rest = math.ceil(seq_len / f)
        max_seq_len = f * rest
        super().__init__(img_size=img_size, patch_size=patch_size, class_token=class_token, qkv_bias=qkv_bias,
                         block_fn=partial(FactorizedDenseBlock, max_seq_len=max_seq_len, f=f), **kwargs)
        self.max_seq_len = max_seq_len
        self.f = f
        self.qkv_bias = qkv_bias

    def set_image_res(self, res):
        if res == self.img_size:
            return
        super().set_image_res(res)

        new_seq_len = (res / self.patch_size) ** 2 + self.num_prefix_tokens
        if not self.keep_f:
            new_f = round(math.sqrt(new_seq_len))
        else:
            new_f = self.f
        rest = math.ceil(new_seq_len / new_f)
        new_max_seq_len = new_f * rest
        old_f = self.f
        old_max_seq_len = self.max_seq_len
        print(f"Changing resolutions: f: {old_f} -> {new_f}, max_seq_len: {old_max_seq_len} -> {new_max_seq_len}")
        self.max_seq_len = new_max_seq_len
        self.f = new_f

        for block in self.blocks:
            num_heads = block.attn.num_heads
            dim = block.attn.dim
            f_a_w = F.interpolate(block.attn.f_a.weight.view(1, num_heads*old_f, dim).transpose(1, 2), size=num_heads*new_f,
                                  mode='linear', align_corners=False).view(dim, num_heads * new_f).transpose(0, 1)
            f_b_w = F.interpolate(block.attn.f_b.weight.view(1, num_heads * old_max_seq_len // old_f, dim).transpose(1, 2),
                                  size=num_heads * new_max_seq_len // new_f, mode='linear', align_corners=False)\
                .view(dim, num_heads * new_max_seq_len // new_f).transpose(0, 1)
            block.attn.f_a.out_features = num_heads * new_f
            block.attn.f_b.out_features = num_heads * new_max_seq_len // new_f
            block.attn.f_a.weight = nn.Parameter(f_a_w.contiguous())
            block.attn.f_b.weight = nn.Parameter(f_b_w.contiguous())
            if self.qkv_bias:
                # rescale bias terms aswell
                f_a_b = F.interpolate(block.attn.f_a.bias.view(1, 1, num_heads * old_f), size=num_heads * new_f,
                                      mode='linear', align_corners=False).view(num_heads * new_f)
                f_b_b = F.interpolate(block.attn.f_b.bias.view(1, 1, num_heads * old_max_seq_len // old_f),
                                      size=num_heads * new_max_seq_len // new_f, mode='linear', align_corners=False)\
                    .view(num_heads * new_max_seq_len // new_f)
                block.attn.f_a.bias = nn.Parameter(f_a_b.contiguous())
                block.attn.f_b.bias = nn.Parameter(f_b_b.contiguous())

            block.attn.max_seq_len = new_max_seq_len
            block.attn.f = new_f


class FactorizedRandomAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len, f, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.random_attn_1 = nn.Parameter(torch.randn(num_heads, max_seq_len, f))
        self.random_attn_2 = nn.Parameter(torch.randn(num_heads, f, max_seq_len))
        self.dropout = nn.Dropout(attn_drop)
        self.num_heads = num_heads
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        random_attn = self.random_attn_1 @ self.random_attn_2  # H x ms x ms
        random_attn = random_attn[:, :N, :N]  # H x N x N
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # B x H x N x d

        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)

        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = einsum('hnm,bhmd->bhnd', random_attn, v)
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class FactorizedRandomBlock(Block):
    def __init__(self, dim, num_heads, max_seq_len, f, qkv_bias=False, drop=0., attn_drop=0., **kwargs):
        super().__init__(dim, num_heads, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, **kwargs)
        self.attn = FactorizedRandomAttention(dim, num_heads, max_seq_len, f, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                              proj_drop=drop)


class FRSynthesizer(TimmViT):
    def __init__(self, img_size=224, patch_size=16, f=8, class_token=True, qkv_bias=False, **kwargs):
        seq_len = (img_size / patch_size) ** 2 + (1 if class_token else 0)
        rest = math.ceil(seq_len / f)
        max_seq_len = f * rest
        super().__init__(img_size=img_size, patch_size=patch_size, class_token=class_token, qkv_bias=qkv_bias,
                         block_fn=partial(FactorizedRandomBlock, max_seq_len=max_seq_len, f=f), **kwargs)
        self.max_seq_len = max_seq_len
        self.f = f

    def set_image_res(self, res):
        if res == self.img_size:
            return
        super().set_image_res(res)

        new_seq_len = (res / self.patch_size) ** 2 + self.num_prefix_tokens
        rest = math.ceil(new_seq_len / self.f)
        new_max_seq_len = self.f * rest
        old_max_seq_len = self.max_seq_len
        print(f"Changing resolutions: max_seq_len: {old_max_seq_len} -> {new_max_seq_len}")
        self.max_seq_len = new_max_seq_len

        for block in self.blocks:
            r_1 = F.interpolate(block.attn.random_attn_1.transpose(1, 2), size=new_max_seq_len, mode='linear',
                                align_corners=False).transpose(1, 2)
            r_2 = F.interpolate(block.attn.random_attn_2, size=new_max_seq_len, mode='linear', align_corners=False)
            block.attn.random_attn_1 = nn.Parameter(r_1.contiguous())
            block.attn.random_attn_2 = nn.Parameter(r_2.contiguous())


@register_model
def synthesizer_fd_vit_tiny_patch16(pretrained=False, img_size=224, f=None, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = FDSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fr_vit_tiny_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = FRSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fd_vit_small_patch16(pretrained=False, img_size=224, f=None, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = FDSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fr_vit_small_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = FRSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fd_vit_base_patch16(pretrained=False, img_size=224, f=None, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = FDSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fr_vit_base_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = FRSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fd_vit_large_patch16(pretrained=False, img_size=224, f=None, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = FDSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model

@register_model
def synthesizer_fr_vit_large_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = FRSynthesizer(img_size=img_size, patch_size=16, in_chans=3, f=f, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          **sizes, **kwargs)
    return model


