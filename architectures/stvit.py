# taken from https://github.com/changsn/STViT-R/blob/main/models/swin_transformer.py with slight modifications
import math
from functools import partial

import torch
from timm.models import register_model
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint
from torch.nn import functional as F
from architectures.swin import Mlp, WindowAttention, window_partition, window_reverse, PatchMerging, PatchEmbed
from resizing_interface import ResizingInterface


class multi_scale_semantic_token1(nn.Module):
    def __init__(self, sample_window_size):
        super().__init__()
        self.sample_window_size = sample_window_size
        self.num_samples = sample_window_size * sample_window_size

    def forward(self, x):
        B, C, _, _ = x.size()
        pool_x = F.adaptive_max_pool2d(x, (self.sample_window_size, self.sample_window_size)).view(B, C, self.num_samples).transpose(2, 1)
        return pool_x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5, use_layer_scale=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        if self.use_layer_scale:
            x = shortcut + self.drop_path(self.layer_scale_1 * x)
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_layer_scale=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_layer_scale=use_layer_scale)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., relative_pos=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.relative_pos = relative_pos
        if self.relative_pos:
            self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size)
            coords_w = torch.arange(self.window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size - 1
            relative_coords[:, :, 0] *= 2 * self.window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, y):
        B, N1, C = x.shape
        B, N2, C = y.shape
        q = self.q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, H)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class SemanticAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, multi_scale, window_size=7, sample_window_size=3, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5,
                 use_conv_pos=False, shortcut=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.multi_scale = multi_scale(sample_window_size)
        self.attn = Attention(dim, num_heads=num_heads, window_size=None, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.use_conv_pos = use_conv_pos
        if self.use_conv_pos:
            self.conv_pos = PosCNN(dim, dim)
        self.shortcut = shortcut
        self.window_size = window_size
        self.sample_window_size = sample_window_size

    def forward(self, x, y=None):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        if y == None:
            xx = x.reshape(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
            windows = xx.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, self.window_size, self.window_size, C).permute(0, 3, 1, 2)
            shortcut = self.multi_scale(windows)  # B*nW, W*W, C
            if self.use_conv_pos:
                shortcut = self.conv_pos(shortcut)
            pool_x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.multi_scale.num_samples, C)
        else:
            B, L_, C = y.shape
            H_ = W_ = int(math.sqrt(L_))
            y = y.reshape(B, H_ // self.sample_window_size, self.sample_window_size, W_ // self.sample_window_size, self.sample_window_size, C)
            y = y.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.sample_window_size*self.sample_window_size, C)
            shortcut = y
            if self.use_conv_pos:
                shortcut = self.conv_pos(shortcut)
            pool_x = self.norm1(shortcut.reshape(B, -1, C)).reshape(-1, self.multi_scale.num_samples, C)
        # produce K, V
        k_windows = F.unfold(x.permute(0, 3, 1, 2), kernel_size=10, stride=4).view(B, C, 10, 10, -1).permute(0, 4, 2, 3, 1)
        k_windows = k_windows.reshape(-1, 100, C)
        k_windows = torch.cat([shortcut, k_windows], dim=1)
        k_windows = self.norm1(k_windows.reshape(B, -1, C)).reshape(-1, 100+self.multi_scale.num_samples, C)

        if self.shortcut:
            x = shortcut + self.drop_path(self.layer_scale_1 * self.attn(pool_x, k_windows))
        else:
            x = self.layer_scale_1 * self.attn(pool_x, k_windows)
        x = x.view(B, H // self.window_size, W // self.window_size, self.sample_window_size, self.sample_window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, C)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class STViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=3, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5, relative_pos=False, local=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.local = local
        self.window_size = window_size

    def forward(self, x, y=None):
        if self.local and y != None:
            raise Exception()
        shortcut = x  # B, L, C
        if y == None:
            y = x
        x = self.norm1(x)
        if self.local:
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size*self.window_size, C)
            attn = self.attn(x, x)
            attn = attn.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
            attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, C)
        else:
            attn = self.attn(x, self.norm1(y))
        x = shortcut + self.drop_path(self.layer_scale_1 * attn)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class STViTDeit(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, input_resolution, window_size, sample_window_size, multi_scale, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', fuse_loc=0, split_loc=2, semantic_key_concat=False,
                 downsample=None, relative_pos=False, use_conv_pos=False, shortcut=False, use_global=True):
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
        self.multi_scale = multi_scale
        num_windows = (input_resolution[0] // window_size) * (input_resolution[1] // window_size)
        self.num_samples = sample_window_size * sample_window_size * num_windows
        self.fuse_loc = fuse_loc
        self.split_loc = split_loc
        self.semantic_key_concat = semantic_key_concat
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.use_global = use_global
        if self.use_global:
            self.semantic_token2 = nn.Parameter(torch.zeros(1, self.num_samples, embed_dim))
            trunc_normal_(self.semantic_token2, std=.02)
        # self.up_dim = nn.Linear(self.embed_dim, 2*self.embed_dim)
        # self.semantic_token_ = nn.Parameter(torch.zeros(1, self.num_samples, embed_dim))
        # trunc_normal_(self.semantic_token_, std=.02)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in [0, 6, 12]:
                self.blocks.append(SwinTransformerBlock(dim=embed_dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i],
                    norm_layer=norm_layer)
                    )
            elif i in [1, 7, 13]:
                self.blocks.append(SemanticAttentionBlock(
                    dim=embed_dim, window_size=window_size, sample_window_size=sample_window_size,
                    num_heads=num_heads, multi_scale=self.multi_scale, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i],
                    norm_layer=norm_layer, act_layer=act_layer,
                    use_conv_pos=use_conv_pos, shortcut=shortcut)
                    )
            elif i in [2, 5, 8, 11, 14, 17]:
                self.blocks.append(STViTBlock(
                    dim=embed_dim, window_size=None, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    drop=drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer, relative_pos=False, local=False)
                    )
            else:
                local = bool(i%2)
                if local:
                    self.blocks.append(STViTBlock(
                        dim=embed_dim, window_size=sample_window_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                        drop=drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer, relative_pos=relative_pos, local=local)
                        )
                else:
                    self.blocks.append(STViTBlock(
                        dim=embed_dim, window_size=input_resolution[0]//window_size*sample_window_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                        drop=drop_rate, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer, relative_pos=relative_pos, local=local)
                        )
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=self.embed_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x)
            elif i == 1:
                semantic_token = blk(x)
            elif i == 2:
                if self.use_global:
                    semantic_token = blk(semantic_token+self.semantic_token2, torch.cat([semantic_token, x], dim=1))
                else:
                    semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 2 and i < 5:
                semantic_token = blk(semantic_token)
            elif i == 5:
                x = blk(x, semantic_token)
            elif i == 6:
                x = blk(x)
            elif i == 7:
                semantic_token = blk(x)
            elif i == 8:
                semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 8 and i < 11:
                semantic_token = blk(semantic_token)
            elif i == 11:
                x = blk(x, semantic_token)
            elif i == 12:
                x = blk(x)
            elif i == 13:
                semantic_token = blk(x)
            elif i == 14:
                semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 14 and i < 17:
                semantic_token = blk(semantic_token)
            else:
                x = blk(x, semantic_token)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        return flops


class STViTSwinTransformer(nn.Module, ResizingInterface):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, multi_scale='multi_scale_semantic_token1',
                 relative_pos=False, use_conv_pos=False, shortcut=False, sample_window_size=3,
                 use_layer_scale=False, use_global=True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.multi_scale = eval(multi_scale)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer in [0, 1, 3]:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                        patches_resolution[1] // (2 ** i_layer)),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    use_layer_scale=use_layer_scale)

            elif i_layer ==2:
                layer = STViTDeit(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                  embed_dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  sample_window_size=sample_window_size,
                                  mlp_ratio=self.mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                  drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                  multi_scale=self.multi_scale,
                                  relative_pos=relative_pos,
                                  use_conv_pos=use_conv_pos,
                                  shortcut=shortcut,
                                  use_global=use_global
                                  )
            # elif i_layer ==3:
            #     layer = Deit2(input_resolution=(patches_resolution[0] // (2 ** i_layer),
            #                                 patches_resolution[1] // (2 ** i_layer)),
            #                 embed_dim=int(embed_dim * 2 ** i_layer),
            #                 depth=depths[i_layer],
            #                 num_heads=num_heads[i_layer],
            #                 num_samples=self.num_samples,
            #                 mlp_ratio=self.mlp_ratio,
            #                 qkv_bias=qkv_bias,
            #                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            #                 drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #                 norm_layer=norm_layer,
            #                 downsample=None
            #     )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.embed_dim = self.num_features
        self.num_classes = num_classes

        self.apply(self._init_weights)

    def set_image_res(self, res):
        if res == self.img_size:
            return
        raise NotImplemented("Resizing the STViT-R-Swin is not implemented, yet.")

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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # for layer in self.layers:
        #     x = layer(x)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


@register_model
def stvit_swin_tiny_p4_w7(img_size=224, pretrained=False, **kwargs):
    if 'layer_scale' in kwargs:
        kwargs['use_layer_scale'] = kwargs['layer_scale']
    model = STViTSwinTransformer(img_size=img_size, patch_size=4, in_chans=3, window_size=7,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=96, depths=[2, 2, 6, 2],
                                 num_heads=[3, 6, 12, 24], **kwargs)
    return model

@register_model
def stvit_swin_base_p4_w7(img_size=224, pretrained=False, **kwargs):
    if 'layer_scale' in kwargs:
        kwargs['use_layer_scale'] = kwargs['layer_scale']
    model = STViTSwinTransformer(img_size=img_size, patch_size=4, in_chans=3, window_size=7,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=192, depths=[2, 2, 18, 2],
                                 num_heads=[6, 12, 24, 48], **kwargs)
    return model
