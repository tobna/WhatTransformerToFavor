# taken from https://github.com/NVlabs/AFNO-transformer/blob/master/classification/afnonet.py with slight changes

import math
import logging
from functools import partial
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from architectures.vit import TimmViT
from resizing_interface import vit_sizes
from timm.models.registry import register_model  # used in exec at the bottom # noqa: F401


_logger = logging.getLogger(__name__)


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(
        self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o1_imag = torch.zeros(
            [B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :, :kept_modes], self.w2[1])
            + self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x


class AFNOBlock(Block):
    def __init__(self, dim, num_heads, blocks, sparsity_threshold=0.01, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.attn = AFNO2D(hidden_size=dim, num_blocks=blocks, sparsity_threshold=sparsity_threshold)


class AFNONet(TimmViT):
    def __init__(self, *args, blocks=8, **kwargs):
        super().__init__(
            *args, **kwargs, block_fn=partial(AFNOBlock, blocks=blocks), global_pool="avg", class_token=False
        )


# generate model creation functions
for size in vit_sizes:
    exec(
        f"""@register_model\n
def afno_{size.lower()}_p16(img_size=224, **kwargs):\n
    sizes = vit_sizes["{size}"]\n
    return AFNONet(img_size=img_size, patch_size=16, in_chans=3, blocks=8, **{{**sizes, **kwargs}})"""
    )
