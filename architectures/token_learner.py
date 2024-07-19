from functools import partial

import torch
from timm.models import register_model
from torch import nn
from timm.models.layers import Mlp

from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.
    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses a MLP with gelu inbetween. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    Based on https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py.
    The original code is licensed under the Apache License, Version 2.0 (see licenses/APACHE_2.0.txt).
    """

    def __init__(self, num_tokens, dim, bottleneck_dim=None, num_prefix_tokens=0, drop=0.0, norm_layer=nn.LayerNorm):
        """

        Parameters
        ----------
        num_tokens : int
            number of tokens to fuse into
        dim : int
            dimension of a token
        bottleneck_dim : int
            intermediate layer width of MLP
        drop : float
            MLP dropout rate
        norm_layer
            normalization layer constructor

        Returns
        -------

        """
        super().__init__()
        self.num_tokens = num_tokens
        self.num_prefix_tokens = num_prefix_tokens

        self.norm = norm_layer(dim)
        bottleneck_dim = bottleneck_dim if bottleneck_dim else dim
        self.mlp = Mlp(dim, bottleneck_dim, num_tokens, drop=drop)

    def forward(self, x):
        """Applies learnable tokenization to the inputs.
        Args:
            x: Inputs of shape `[B, N, C]`.
        Returns:
            Output of shape `[B, num_prefix_tokens + num_tokens, C]`.
        """

        B, N, C = x.shape
        prefix_tokens = x[:, : self.num_prefix_tokens]
        x = x[:, self.num_prefix_tokens :]

        selected = self.norm(x)  # B x N x C
        selected = self.mlp(selected)  # B x N x tokens
        selected = selected.transpose(1, 2).softmax(dim=-1)  # B x tokens x N (norm)

        out = selected @ x  # B x tokens x C
        out = torch.cat((prefix_tokens, out), dim=1)

        return out


class TokenLearnerViT(TimmViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        token_learner_layer_rel=0.5,
        learned_tokens=8,
        embed_dim=768,
        mlp_ratio=4.0,
        drop_rate=0.0,
        norm_layer=None,
        depth=12,
        **kwargs
    ):
        norm_layer = norm_layer if norm_layer is not None else partial(nn.LayerNorm, eps=1e-6)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
            depth=depth,
            **kwargs
        )
        token_learner = TokenLearnerModuleV11(
            learned_tokens,
            embed_dim,
            bottleneck_dim=int(mlp_ratio * embed_dim),
            num_prefix_tokens=self.num_prefix_tokens,
            drop=drop_rate,
            norm_layer=norm_layer,
        )
        self.token_learner_layer = int(token_learner_layer_rel * depth)
        block_list = list(self.blocks)
        block_list = block_list[: self.token_learner_layer] + [token_learner] + block_list[: self.token_learner_layer]
        self.blocks = nn.Sequential(*block_list)


@register_model
def token_learner_vit_8_50_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_75_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.75,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_50_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_75_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.75,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_50_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_75_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.75,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_50_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model


@register_model
def token_learner_vit_8_75_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = TokenLearnerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        learned_tokens=8,
        token_learner_layer_rel=0.75,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs
    )
    return model
