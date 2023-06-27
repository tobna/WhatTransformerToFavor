from functools import partial

import torch
from timm.models import register_model
from torch import nn
from architectures.vit import TimmViT
from timm.models.vision_transformer import Block

from resizing_interface import vit_sizes


class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft2(x, dim=(-1, -2), norm='forward').real
        return x


class FNetBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., proj_drop=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, proj_drop=proj_drop, init_values=init_values, act_layer=act_layer, norm_layer=norm_layer)
        assert num_heads == 1, f"FNet can only work with one head, but got {num_heads}"
        self.attn = FFT()

@register_model
def fnet_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["B"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["L"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_tiny_patch4(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=4, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_small_patch4(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=4, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_base_patch4(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["B"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=4, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model

@register_model
def fnet_vit_large_patch4(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["L"]
    sizes['num_heads'] = 1
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = TimmViT(img_size=img_size, patch_size=4, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='avg', block_fn=FNetBlock, **kwargs, **sizes)
    return model
