# taken from https://github.com/NVlabs/A-ViT/blob/master/timm/models/act_vision_transformer.py with slight modifications
# The original code is licensed under the Apache License, Version 2.0 (see licenses/APACHE_2.0.txt)

# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

# The following snippets are started from:
# https://github.com/facebookresearch/deit
# &
# https://github.com/rwightman/pytorch-image-models
# Before code is extensively modified to accomodate A-ViT training

import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from timm.models import register_model
from torch import nn
from timm.models.layers import DropPath, Mlp, PatchEmbed, lecun_normal_
from torch.autograd import Variable
from torch.nn.init import trunc_normal_

from resizing_interface import ResizingInterface, vit_sizes
from utils import DotDict


class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None,
                 masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask  # this is of shape [batch, token_number], where the token number
        # dimension is indication of token exec.
        # 0's are the tokens to continue, 1's are the tokens masked out

        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # now we need to mask out all the attentions associated with this token
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias
            # this additional bias will make attention associated with this token to be zeroed out
            # this incurs at each head, making sure all embedding sections of other tokens ignore these tokens

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_ACT(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4}  # now only support 1-extra mlp, or b-position 0 encoding

        self.index = index
        self.args = args

        if self.act_mode == 4:
            # Apply sigmoid on the mean of all tokens to determine whether to continue
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_act(self, x, mask=None):

        debug = False
        analyze_delta = True
        bs, token, dim = x.shape

        if mask is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.attn(self.norm1(x * (1 - mask).view(bs, token, 1)) * (1 - mask).view(bs, token, 1), mask=mask))
            x = x + self.drop_path(
                self.mlp(self.norm2(x * (1 - mask).view(bs, token, 1)) * (1 - mask).view(bs, token, 1)))

        if self.act_mode == 4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:, :, 0] * gate_scale - gate_center)
            # initially first position used for layer halting, second for token
            # now discarding position 1
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score


def get_distribution_target(mode='gaussian', length=12, max=1, standardized=True, target_depth=8, buffer=0.02):
    """
    This generates the target distributional prior
    """
    # this gets the distributional target to regularize the ACT halting scores towards
    if mode == 'gaussian':
        from scipy.stats import norm
        # now get a serios of length
        data = np.arange(length)
        data = norm.pdf(data, loc=target_depth, scale=1)

        if standardized:
            print('\nReshaping distribution to be top-1 sum 1 - error at {}'.format(buffer))
            scaling_factor = (1. - buffer) / sum(data[:target_depth])
            data *= scaling_factor

        return data

    elif mode == 'lognorm':
        from scipy.stats import lognorm

        data = np.arange(length)
        data = lognorm.pdf(data, s=0.99)

        if standardized:
            print('\nReshaping distribution to be top-1 sum 1 - error at {}'.format(buffer))
            scaling_factor = (1. - buffer) / sum(data[:target_depth])
            data *= scaling_factor

        print('\nForming distribution at:', data)
        return data

    elif mode == 'skewnorm':
        from scipy.stats import skewnorm
        # now get a serios of length
        data = np.arange(1, length)
        data = skewnorm.pdf(data, a=-4, loc=target_depth)
        return data

    else:
        print('Get distributional prior not implemented!')
        raise NotImplementedError


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# Adaptive Vision Transformer
class AViT(nn.Module, ResizingInterface):
    """ Vision Transformer with Adaptive Token Capability
    Starting at:
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
    Extended to:
        Accomodate adaptive token inference
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None, **kwargs):
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
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 2 if distilled else 1
        self.no_embed_class = False
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.embed_layer = embed_layer
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pre_norm = False

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block_ACT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args,
                index=i, num_patches=self.patch_embed.num_patches + 1)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

        print('\nNow this is an ACT DeiT.\n')
        self.eps = 0.01
        print(f'Setting eps as {self.eps}.')

        print('Now re-initializing the halting network bias')
        for block in self.blocks:
            if args.act_mode == 1:
                # torch.nn.init.constant_(block.act_mlp.fc1.bias.data, -3)
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        print('Now setting up the rho.')
        self.rho = None  # Ponder cost
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0  # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.halting_score_layer = None
        self.total_token_cnt = num_patches + self.num_prefix_tokens

        if args.distr_prior_alpha > 0.:
            self.distr_target = torch.Tensor(get_distribution_target(standardized=True)).cuda()
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    def get_internal_loss(self):
        ponder_loss_token = self.rho_token.view(-1).mean(0) * self.args.ponder_token_scale if self.rho_token is not None else 0.
        halting_score_dist = torch.stack(self.halting_score_layer)
        halting_score_dist = halting_score_dist / halting_score_dist.sum()
        halting_score_dist = halting_score_dist.clamp(min=0.01, max=0.99)
        distr_prior_loss = self.args.distr_prior_alpha * self.kl_loss(halting_score_dist.log(), self.distr_target)
        return ponder_loss_token + distr_prior_loss

    def set_image_res(self, res):
        if self.img_size == res:
            return
        super().set_image_res(res)
        self.c_token = self.R_token = self.mask_token = self.rho_token = self.counter_token = None
        self.total_token_cnt = self.patch_embed.num_patches + self.num_prefix_tokens

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_prefix_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features_act_token(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # now start the act part
        bs = x.size()[0]  # The batch size

        # this part needs to be modified for higher GPU utilization
        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = torch.zeros_like(self.rho_token, requires_grad=not self.training)  # self.rho_token.detach() * 0.
        self.counter_token = torch.zeros_like(self.counter_token, requires_grad=not self.training) + 1.  # self.counter_token.detach() * 0 + 1.
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x

        if self.args.distr_prior_alpha > 0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):

            # block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1)

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = l.forward_act(out,
                                                1. - mask_token.float())  # h is a vector of length bs, block_output a 3D tensor

            if self.args.distr_prior_alpha > 0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()  # Deep copy needed for the next layer

            _, h_token = h_lst  # h is layer_halting score, h_token is token halting score, first position discarded

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            # Is this the last layer in the block?
            if i == len(self.blocks) - 1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

            # for token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs,
                                                                                                   self.total_token_cnt,
                                                                                                   1)
            self.rho_token = self.rho_token + R_token * reached_token

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs,
                                                                                                       self.total_token_cnt,
                                                                                                       1)

            self.counter_token = self.counter_token + not_reached_token  # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward_probs(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        out_lst = []
        assert self.dist_token is None

        for i, l in enumerate(self.blocks):
            # evaluate layer and get halting probability for each sample
            out = l.forward(x)  # h is a vector of length bs, block_output a 3D tensor
            tmp_prob = self.head(self.pre_logits(self.norm(out)[:, 0]))
            out_lst.append(tmp_prob)
            x = out

        return out_lst

    def forward(self, x):
        if self.args.act_mode == 4:
            x = self.forward_features_act_token(x)
        else:
            print('Not implemented yet, please specify for token act.')
            exit()

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return x, rho, count # discarded from v1
        return x


@register_model
def avit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["Ti"]
    model = AViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes,
                 args=DotDict(dict(act_mode=4, distr_prior_alpha=.01, gate_scale=100., gate_center=3.,
                                   ponder_token_scale=0.001)), **kwargs)
    return model

@register_model
def avit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["S"]
    model = AViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes,
                 args=DotDict(dict(act_mode=4, distr_prior_alpha=.01, gate_scale=100., gate_center=3.,
                                   ponder_token_scale=0.001)), **kwargs)
    return model

@register_model
def avit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["B"]
    model = AViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes,
                 args=DotDict(dict(act_mode=4, distr_prior_alpha=.01, gate_scale=100., gate_center=3.,
                                   ponder_token_scale=0.001)), **kwargs)
    return model

@register_model
def avit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if 'layer_scale_init_values' in kwargs:
        kwargs['init_values'] = kwargs['layer_scale_init_values'] if 'layer_scale' in kwargs and kwargs['layer_scale'] else None
    sizes = vit_sizes["L"]
    model = AViT(img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes,
                 args=DotDict(dict(act_mode=4, distr_prior_alpha=.01, gate_scale=100., gate_center=3.,
                                   ponder_token_scale=0.001)), **kwargs)
    return model


