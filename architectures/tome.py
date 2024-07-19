# Taken from https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py with slight modifications.
# The original code is licensed under the CC BY-NC: Attribution-NonCommercial 4.0 license (see licenses/CC-BY-NC_4.0.txt).

import math
from functools import partial

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY-NC: Attribution-NonCommercial 4.0 license (see licenses/CC-BY-NC_4.0.txt).
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple, Union, List, Callable

import torch
from timm.models import register_model
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from torch import nn

from architectures.vit import TimmViT
from resizing_interface import vit_sizes


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(metric, r, class_token, distill_token):
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Parameters
    ----------
    metric : torch.Tensor
        The input tensor of shape [batch, tokens, channels] to be processed.
    r : int
        The number of tokens to remove, with a maximum of 50% of tokens.
    class_token : bool, optional
        Whether the input tensor has a class token, by default False.
    distill_token : bool, optional
        Whether the input tensor has a distillation token, by default False.

    Returns
    -------
    Tuple[Callable, Callable]
        A tuple of two callables.

    Notes
    -----
    When `class_token` and/or `distill_token` are enabled, the class token and/or
    distillation token(s) will not be merged.
    """

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant `r` or `r` schedule into a list for use internally.

    Parameters
    ----------
    num_layers : int
        The number of layers in the model.
    r : Union[List[int], Tuple[int, float], int]
        `r` can take the following forms:

        - `int`: A constant number of tokens per layer.
        - `Tuple[int, float]`: A pair of `r`, `inflection`.
          `Inflection` describes where the reduction/layer should trend
          upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
          is as providing a constant r. (r, -1) is what we describe in the paper
          as "decreasing schedule". Any value between -1 and +1 is accepted.
        - `List[int]`: A specific number of tokens per layer. For extreme granularity.

    Returns
    -------
    List[int]
        A list of `r` values per layer.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the given merge function by taking a weighted average based on token size.

    Parameters
    ----------
    merge : Callable
        A function to merge the tensors.
    x : torch.Tensor
        The tensor to merge.
    size : torch.Tensor, optional
        A tensor of token sizes. If not given, each token is assumed to have a size of 1.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The merged tensor and the new token sizes.
    """

    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """
    Merge the tensor x using the merge function while keeping track of its source.
    Returns the merged tensor.

    Parameters
    ----------
    merge : Callable
        The function used for merging the tensor.
    x : torch.Tensor
        The tensor to be merged.
    source : torch.Tensor, optional
        An adjacency matrix between the initial tokens and final merged groups.
        If None, x is used to determine the number of tokens.

    Returns
    -------
    torch.Tensor
        The merged tensor.

    Notes
    -----
    This function is used for source tracking. If the source is None, x is used to determine
    the number of tokens in order to create an identity source matrix.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(self.ls1(x_attn))

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ToMeAttention(Attention):
    """
    Modified Attention module that applies proportional attention and returns the mean of k over heads.

    Modifications:
     - Applies proportional attention based on token size
     - Returns the mean of k over heads from attention
    """

    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the modified attention mechanism to the input tensor x.
        Returns a tuple containing the attended output tensor and the mean of k over heads.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_size).
            size (torch.Tensor): The tensor containing the size information of tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attended output tensor and the mean of k over heads.
        """
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to the given transformer model and sets the value of r using model.r.

    Parameters
    ----------
    model : VisionTransformer
        The transformer model to which ToMe is to be applied.
    trace_source : bool, optional
        A flag indicating whether to trace the source of each token for visualization purposes.
        Defaults to False.
    prop_attn : bool, optional
        A flag indicating whether to apply proportional attention. This is necessary only when
        evaluating off-the-shelf models. Defaults to True.

    Notes
    -----
    After applying ToMe, the sources of each token can be accessed at model._tome_info["source"].

    When evaluating MAE models off-the-shelf, it is recommended to set prop_attn to False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention


@register_model
def tome_vit_tiny_r8_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 8
    return model


@register_model
def tome_vit_tiny_r13_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 13
    return model


@register_model
def tome_vit_small_r8_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 8
    return model


@register_model
def tome_vit_small_r13_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 13
    return model


@register_model
def tome_vit_base_r8_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 8
    return model


@register_model
def tome_vit_base_r13_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 13
    return model


@register_model
def tome_vit_large_r8_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 8
    return model


@register_model
def tome_vit_large_r13_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = TimmViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    apply_patch(model)
    model.r = 13
    return model
