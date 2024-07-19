import os

import torch
from timm.models import register_model
from timm.models.cait import Cait as CaitTimm
from timm.models.layers import PatchEmbed

from resizing_interface import ResizingInterface

_CAIT_ARGS = [
    "img_size",
    "patch_size",
    "in_chans",
    "num_classes",
    "global_pool",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "qkv_bias",
    "drop_rate",
    "attn_drop_rate",
    "drop_path_rate",
    "block_layers",
    "block_layers_token",
    "patch_layer",
    "norm_layer",
    "act_layer",
    "attn_block",
    "mlp_block",
    "init_values",
    "attn_block_token_only",
    "mlp_block_token_only",
    "depth_token_only",
    "mlp_ratio_token_only",
]


def _force_cudnn_init(s=32):
    if "LOCAL_RANK" in os.environ:
        dev = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    else:
        dev = torch.device("cuda")
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


class CaiT(CaitTimm, ResizingInterface):
    def __init__(self, img_size=224, patch_layer=PatchEmbed, patch_size=16, in_chans=3, **kwargs):
        for key in list(kwargs.keys()):
            if key not in _CAIT_ARGS:
                kwargs.pop(key)
        super().__init__(patch_layer=patch_layer, patch_size=16, in_chans=3, img_size=img_size, **kwargs)
        self.embed_layer = patch_layer
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pre_norm = False
        self.no_embed_class = True
        self.img_size = img_size
        _force_cudnn_init()


@register_model
def cait_xxs24(pretrained=False, img_size=224, init_values=1e-5, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_xxs36(pretrained=False, img_size=224, init_values=1e-5, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=192, depth=36, num_heads=4, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_xs24(pretrained=False, img_size=224, init_values=1e-5, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=288, depth=24, num_heads=6, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_s24(pretrained=False, img_size=224, init_values=1e-5, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_s36(pretrained=False, img_size=224, init_values=1e-6, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=384, depth=36, num_heads=8, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_m36(pretrained=False, img_size=224, init_values=1e-6, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=768, depth=36, num_heads=16, init_values=init_values, img_size=img_size, **kwargs
    )
    return model


@register_model
def cait_m48(pretrained=False, img_size=224, init_values=1e-6, **kwargs):
    model = CaiT(
        patch_size=16, embed_dim=768, depth=48, num_heads=16, init_values=init_values, img_size=img_size, **kwargs
    )
    return model
