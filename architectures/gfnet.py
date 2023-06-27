# Taken from https://github.com/raoyongming/GFNet with slight modifications

import logging
import math
from collections import OrderedDict
from copy import copy
from functools import partial
from timm.models import register_model
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath
from resizing_interface import ResizingInterface


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

    def reshape(self, width, height):
        if width == self.w and height == self.h:
            return

        # comlpex_weight has shape h x w x dim x 2
        old_shape = self.complex_weight.shape
        weight = self.complex_weight.permute(3, 2, 1, 0)  # 2 x dim x w x h
        weight = F.interpolate(weight, size=(width, height), mode='bilinear')  # 2 x dim x width x height
        self.complex_weight = nn.Parameter(weight.permute(3, 2, 1, 0).contiguous())  # height x width x dim x 2
        self.w = width
        self.h = height


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class GFNet(nn.Module, ResizingInterface):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0, **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.embed_layer = PatchEmbed
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print(f'using uniform droppath with expect rate {drop_path_rate}')
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            drop_path_rate *= 2
            print(f'using linear droppath with expect rate {drop_path_rate * 0.5}')
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            logging.info('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def set_image_res(self, res):
        if res == self.img_size:
            return

        old_patch_embed_state = copy(self.patch_embed.state_dict())
        patch_size = self.patch_embed.patch_size
        self.patch_embed = self.embed_layer(
            img_size=res, patch_size=patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed.load_state_dict(old_patch_embed_state)

        pos_embed_new = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_embed = nn.Parameter(resize_pos_embed(self.pos_embed, pos_embed_new))

        # rescale global filters
        assert patch_size[0] == patch_size[1], f"Got patch size {patch_size}"
        h = res // patch_size[0]
        w = h // 2 + 1
        for block in self.blocks:
            block.filter.reshape(w, h)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print(f'Resized position embedding: {posemb.shape} to {posemb_new.shape}')
    ntok_new = posemb_new.shape[1]
    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    print(f'Position embedding grid-size from {gs_old} to {gs_new}')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1).contiguous()
    return posemb


@register_model
def gfnet_tiny_patch4(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=4, in_chans=3, depth=12, embed_dim=256, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_extra_small_patch4(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=4, in_chans=3, depth=12, embed_dim=386, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_small_patch4(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=4, in_chans=3, depth=19, embed_dim=384, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_base_patch4(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=4, in_chans=3, depth=19, embed_dim=512, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=16, in_chans=3, depth=12, embed_dim=256, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_extra_small_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=16, in_chans=3, depth=12, embed_dim=386, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_small_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=16, in_chans=3, depth=19, embed_dim=384, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model

@register_model
def gfnet_base_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    model = GFNet(img_size=img_size, patch_size=16, in_chans=3, depth=19, embed_dim=512, norm_layer=partial(nn.LayerNorm, eps=1e-6), uniform_drop=True, **kwargs)
    return model
