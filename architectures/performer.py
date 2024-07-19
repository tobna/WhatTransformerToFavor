# based on https://github.com/lucidrains/performer-pytorch with slight modifications
# The original code is licensed under the MIT license (see licenses/MIT.txt) from Phil Wang, 2020.

import math
import torch
import torch.nn.functional as F
from timm.models import register_model
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager
from timm.models.vision_transformer import Block
from distutils.version import LooseVersion

from architectures.vit import TimmViT
from resizing_interface import vit_sizes

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


# helpers


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


# token shifting helper and classes


def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return F.pad(t, (0, 0, amount, -amount), value=0.0)


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )

    return data_dash.type_as(data)


def generalized_kernel(
    data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True, device=None
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel


# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
    context = torch.einsum("...nd,...ne->...de", k, v)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
    return out


# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps=1e-6):
    from fast_transformers.causal_product import CausalDotProduct

    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, "half tensors can only be used if nvidia apex is available"
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled=False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum("...nd,...n->...nd", out, D_inv)
    return out


# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q) + eps)
        context = torch.einsum("...nd,...ne->...nde", k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum("...nde,...nd,...n->...ne", context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
        feature_redraw_interval=1000,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.feature_redraw_interval = feature_redraw_interval
        self.steps_since_last_redraw = 0

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print(
                    "unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version"
                )
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.training and self.feature_redraw_interval:
            self.steps_since_last_redraw += 1
            if self.steps_since_last_redraw > self.feature_redraw_interval:
                self.redraw_projection_matrix(device)
                self.steps_since_last_redraw = 0

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(
                generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix, device=device
            )
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out


class PerformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=None,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.0,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection,
            feature_redraw_interval=feature_redraw_interval,
        )

        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):
        h = self.heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        attn_outs = []

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.0)

        if exists(pos_emb) and not cross_attend:
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = self.fast_attention(q, k, v)
        attn_outs.append(out)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class SelfAttention(PerformerAttention):
    def forward(self, *args, context=None, **kwargs):
        assert not exists(context), "self attention should not receive context"
        return super().forward(*args, **kwargs)


class CrossAttention(PerformerAttention):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), "cross attention should receive context"
        return super().forward(*args, context=context, **kwargs)


# rotary positional embedding helpers


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, "() n (j d) -> n j d", j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


class PerformerBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        feature_redraw_interval=1000,
        attn_drop=0.0,
        qkv_bias=False,
        generalized_attention=True,
        kernel_fn=nn.ReLU(),
        nb_features=None,
        **kwargs,
    ):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)
        self.attn = PerformerAttention(
            dim,
            heads=num_heads,
            feature_redraw_interval=feature_redraw_interval,
            dropout=attn_drop,
            qkv_bias=qkv_bias,
            nb_features=nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
        )


class PerformerViT(TimmViT):
    def __init__(
        self, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), nb_features=None, **kwargs
    ):
        super().__init__(
            block_fn=partial(
                PerformerBlock,
                feature_redraw_interval=feature_redraw_interval,
                nb_features=nb_features,
                generalized_attention=generalized_attention,
                kernel_fn=kernel_fn,
            ),
            **kwargs,
        )


# TODO: Try 64/128 features, instead of dim*log(dim)
@register_model
def performer_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = PerformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        feature_redraw_interval=None,
        nb_features=128,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def performer_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = PerformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        feature_redraw_interval=None,
        nb_features=128,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def performer_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = PerformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        feature_redraw_interval=None,
        nb_features=128,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def performer_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = PerformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        feature_redraw_interval=None,
        nb_features=128,
        **sizes,
        **kwargs,
    )
    return model
