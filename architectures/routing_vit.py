import math
from functools import partial

import torch
from routing_transformer.routing_transformer import SelfAttention as LocalGlobalRouting, Kmeans, default, \
    split_at_index, distribution, batched_index_select, expand_dim, exists, max_neg_value, TOKEN_SELF_ATTN_VALUE, \
    scatter_mean
from routing_transformer import routing_transformer
from timm.models import register_model
from timm.models.vision_transformer import Block
from torch import nn
from torch.nn import functional as F

from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class KmeansAttentionDDP(nn.Module):
    # This is a sligtly modified version of the code from https://github.com/lucidrains/routing-transformer
    # The original code is licensed under the MIT license (see licenses/MIT.txt) from Phil Wang, 2020.
    def __init__(self, num_clusters, window_size, num_heads, head_dim, causal=False, dropout=0., ema_decay=0.999,
                 commitment=1e-4, context_window_size=None, receives_context=False, num_mem_kv=0, shared_qk=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim

        self.window_size = window_size
        self.context_window_size = default(context_window_size, window_size)
        self.causal = causal

        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)
        self.dropout = nn.Dropout(dropout)

        self.num_mem_kv = max(num_mem_kv, 1 if causal and not shared_qk else 0)
        if self.num_mem_kv > 0:
            self.mem_key = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
            self.mem_value = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))

    def forward(self, q, k, v, query_mask=None, key_mask=None, **kwargs):
        b, h, t, d, kv_t, wsz, c_wsz, nc, device, dtype = *q.shape, k.shape[
            2], self.window_size, self.context_window_size, self.num_clusters, q.device, q.dtype
        is_reverse = kwargs.pop('_reverse', False)

        out = torch.zeros_like(q, dtype=dtype)

        update_kmeans = self.training and not is_reverse

        key_mask = default(key_mask, query_mask) if not self.receives_context else key_mask
        kv_wsz = wsz if not self.receives_context else c_wsz

        wsz = min(wsz, t)
        kv_wsz = min(kv_wsz, kv_t)

        if not self.shared_qk or self.receives_context:
            dists, aux_loss = self.kmeans(torch.cat((q, k), dim=2), update_kmeans)
            q_dists, k_dists = split_at_index(2, t, dists)
            indices = distribution(q_dists, wsz)
            kv_indices = distribution(k_dists, kv_wsz)
        else:
            dists, aux_loss = self.kmeans(q, update_kmeans)
            k = F.normalize(k, dim=-1).to(q)
            indices = distribution(dists, wsz)
            kv_indices = indices

        q = batched_index_select(q, indices)
        k = batched_index_select(k, kv_indices)
        v = batched_index_select(v, kv_indices)

        reshape_with_window = lambda x: x.reshape(b, h, nc, -1, d)
        q, k, v = map(reshape_with_window, (q, k, v))

        if self.num_mem_kv > 0:
            m_k, m_v = map(lambda x: expand_dim(x, 0, b).to(q), (self.mem_key, self.mem_value))
            k, v = map(lambda x: torch.cat(x, dim=3), ((m_k, k), (m_v, v)))

        dots = torch.einsum('bhnid,bhnjd->bhnij', q, k) * (d ** -0.5)

        mask_value = max_neg_value(dots)

        if exists(query_mask) or exists(key_mask):
            query_mask = default(query_mask, lambda: torch.ones((b, t), device=device).bool())
            key_mask = default(key_mask, lambda: torch.ones((b, kv_t), device=device).bool())

            q_mask = expand_dim(query_mask, 1, h).gather(2, indices)
            kv_mask = expand_dim(key_mask, 1, h).gather(2, kv_indices)
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (q_mask, kv_mask))
            mask = q_mask[:, :, :, :, None] * kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices))
            mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.shared_qk:
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices))
            mask = q_mask[:, :, :, :, None] == kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=False)
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        bo = torch.einsum('bhcij,bhcjd->bhcid', dots, v)
        so = torch.reshape(bo, (b, h, -1, bo.shape[-1])).type(dtype)
        out = scatter_mean(out, so, indices.unsqueeze(-1).expand_as(so), -2)
        return out, aux_loss


routing_transformer.KmeansAttention = KmeansAttentionDDP


class RoutingAttention(LocalGlobalRouting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = 0.

    def forward(self, x, **kwargs):
        out, loss = super().forward(x, **kwargs)
        self.loss += loss
        return out


class RoutingBlock(Block):
    def __init__(self, dim, max_seq_len, num_heads, num_local_heads, window_size, proj_drop=0., attn_drop=0., **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, proj_drop=proj_drop, attn_drop=attn_drop, **kwargs)
        self.attn = RoutingAttention(dim, depth=-1, max_seq_len=max_seq_len, heads=num_heads,
                                     local_attn_heads=num_local_heads, window_size=window_size, causal=False,
                                     attn_dropout=attn_drop, dropout=proj_drop)
        self.loss = 0

    def forward(self, x):
        out = super().forward(x)
        if self.training:
            self.loss += self.attn.loss
        self.attn.loss = 0.
        return out


class RoutingViT(TimmViT):
    def __init__(self, img_size=224, patch_size=16, local_heads_fact=0., num_heads=12, class_token=True, **kwargs):
        assert img_size % patch_size == 0, f"Patch size ({patch_size}) has to be divisible " \
                                           f"by image resolution ({img_size})."
        seq_len = (img_size // patch_size) ** 2 + (1 if class_token else 0)
        local_heads = math.floor(local_heads_fact * num_heads)
        optimal_window_size = math.sqrt(seq_len)
        window_size = round(optimal_window_size)
        rest_fact = math.ceil(seq_len / window_size)
        super().__init__(img_size=img_size, patch_size=patch_size, num_heads=num_heads, class_token=class_token,
                         block_fn=partial(RoutingBlock, max_seq_len=window_size*rest_fact, num_local_heads=local_heads,
                                          window_size=window_size),
                         **kwargs)
        self.window_size = window_size
        self.rest_fact = rest_fact
        self.local_heads = local_heads
        self.local_heads_fact = local_heads_fact
        self.seq_len = seq_len
        self.loss = 0.

    def forward(self, x):
        out = super().forward(x)
        for block in self.blocks:
            self.loss += block.loss
            block.loss = 0.
        return out

    def get_internal_loss(self):
        internal_loss = self.loss
        self.loss = 0.
        return internal_loss
        
    def set_image_res(self, res):
        if res == self.img_size:
            return

        super().set_image_res(res)
        new_seq_len = (res // self.patch_size) ** 2 + self.num_prefix_tokens
        old_seq_len = self.seq_len
        self.seq_len = new_seq_len
        old_max_seq_len = self.window_size * self.rest_fact
        old_window_size = self.window_size
        optimal_window_size = math.sqrt(new_seq_len)
        new_window_size = round(optimal_window_size)
        rest_fact = math.ceil(new_seq_len / new_window_size)
        new_max_seq_len = rest_fact * new_window_size
        self.window_size = new_window_size
        self.rest_fact = rest_fact
        self.seq_len = new_seq_len

        new_num_clusters = new_max_seq_len // new_window_size
        print(f"Resizing: seq_len: {old_seq_len} |-> {new_seq_len}, "
              f"max_seq_len: {old_max_seq_len} |-> {new_max_seq_len}, "
              f"num_clusters: {old_max_seq_len // old_window_size} |-> {new_max_seq_len // new_window_size}")
        for block in self.blocks:
            old_num_clusters = block.attn.global_attn.num_clusters
            block.attn.global_attn.num_clusters = new_num_clusters
            # old_mem_key = block.attn.global_attn.mem_key
            # old_mem_value = block.attn.global_attn.mem_value
            old_means = block.attn.global_attn.kmeans.means

            # if old_num_clusters >= new_num_clusters:
            #     new_mem_key = old_mem_key[:, :new_num_clusters]
            #     new_mem_value = old_mem_value[:, :new_num_clusters]
            #     new_means = old_means[:, :new_num_clusters]
            # else:
            #     new_keys = torch.randn(block.attn.global_attn.num_heads, new_num_clusters - old_num_clusters,
            #                            block.attn.global_attn.num_mem_kv, block.attn.global_attn.head_dim)
            #     new_values = torch.randn(block.attn.global_attn.num_heads, new_num_clusters - old_num_clusters,
            #                              block.attn.global_attn.num_mem_kv, block.attn.global_attn.head_dim)
            #     additional_means = torch.randn(block.attn.global_attn.num_heads, new_num_clusters - old_num_clusters,
            #                             block.attn.global_attn.head_dim)
            #     new_mem_key = torch.cat((old_mem_key, new_keys), dim=1)
            #     new_mem_value = torch.cat((old_mem_value, new_values), dim=1)
            #     new_means = torch.cat((old_means, additional_means), dim=1)

            head_dim = block.attn.global_attn.head_dim
            num_heads = block.attn.global_attn.num_heads
            num_mem_kv = block.attn.global_attn.num_mem_kv
            if num_mem_kv > 0:
                old_mem_key = old_mem_key.permute(0, 3, 2, 1).reshape(head_dim*num_heads, num_mem_kv, old_num_clusters)
                new_mem_key = nn.functional.interpolate(old_mem_key, size=new_num_clusters, mode='linear')\
                    .reshape(num_heads, head_dim, num_mem_kv, new_num_clusters).permute(0, 3, 2, 1)
                old_mem_value = old_mem_value.permute(0, 3, 2, 1).reshape(head_dim*num_heads, num_mem_kv, old_num_clusters)
                new_mem_value = nn.functional.interpolate(old_mem_value, size=new_num_clusters, mode='linear')\
                    .reshape(num_heads, head_dim, num_mem_kv, new_num_clusters).permute(0, 3, 2, 1)
                block.attn.global_attn.mem_key = nn.Parameter(new_mem_key)
                block.attn.global_attn.mem_value = nn.Parameter(new_mem_value)
            # else:
            #     new_mem_key = old_mem_key.reshape(num_heads, new_num_clusters, num_mem_kv, head_dim)
            #     new_mem_value = old_mem_value.reshape(num_heads, new_num_clusters, num_mem_kv, head_dim)
            old_means = old_means.permute(0, 2, 1)
            new_means = nn.functional.interpolate(old_means, size=new_num_clusters, mode='linear')\
                .permute(0, 2, 1)
            block.attn.global_attn.kmeans.register_buffer('means', new_means)


@register_model
def routing_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = RoutingViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       **sizes, **kwargs)
    return model

@register_model
def routing_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = RoutingViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       **sizes, **kwargs)
    return model

@register_model
def routing_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = RoutingViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       **sizes, **kwargs)
    return model

@register_model
def routing_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = RoutingViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       **sizes, **kwargs)
    return model
