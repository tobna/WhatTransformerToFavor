from halonet_pytorch import HaloAttention as HaloAttentionPckg
from timm.models import register_model
from torch import nn

from resizing_interface import ResizingInterface

# The halonet_pytorch package is licensed under the MIT license (see licenses/MIT.txt) from Phil Wang, 2021.


# look into https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/halonet/halonet.py
class HaloAttention(HaloAttentionPckg):
    def __init__(self, dim, block_size, halo_size, dim_head=None, heads=8, dim_out=None, qkv_bias=False):
        if dim_head is None:
            dim_head = dim // heads
        super().__init__(dim=dim, block_size=block_size, halo_size=halo_size, dim_head=dim_head, heads=heads)
        inner_dim = dim_head * heads
        if dim_out is not None:
            self.to_out = nn.Linear(inner_dim, dim_out)

        if qkv_bias:
            self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)


class GlobalAvgPool(nn.Module):
    def __init__(self, dims=(-1, -2)):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)


class Downsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if len(x.shape) == 4:
            return x[:, :, 0 :: self.stride, 0 :: self.stride]
        elif len(x.shape) == 3:
            return x[:, :, 0 :: self.stride]
        raise NotImplementedError(f"Subsampling is not implemented for tensor of shape {x.shape}")


def _create_halo_block(
    base_dim,
    in_dim,
    heads,
    rv,
    rb,
    block_size,
    halo_size,
    activation=nn.GELU(),
    block_depth=3,
    downsampling=True,
    qkv_bias=False,
):
    sequence = [
        [
            nn.Conv2d(in_dim if i == 0 else int(rb * base_dim), base_dim, 1),
            activation,
            HaloAttention(base_dim, block_size, halo_size, heads=heads, dim_out=int(base_dim * rv), qkv_bias=qkv_bias),
            activation,
            nn.Conv2d(int(base_dim * rv), int(base_dim * rb), 1),
            activation,
        ]
        for i in range(block_depth)
    ]
    sequence = [mod for subblck in sequence for mod in subblck]
    if downsampling:
        sequence.append(Downsample())
    return nn.Sequential(*sequence)


class HaloNet(nn.Module, ResizingInterface):
    def __init__(
        self,
        num_classes=1000,
        rv=1.0,
        rb=1.5,
        in_chans=3,
        block_size=8,
        halo_size=3,
        l3=10,
        num_heads=None,
        embed_dim=None,
        activation=nn.GELU(),
        qkv_bias=False,
        **kwargs,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = [4, 8, 8, 8]
        self.start_block = nn.Sequential(
            nn.Conv2d(in_chans, 64, 7, stride=2, padding=3), nn.MaxPool2d(3, 2, padding=1), activation
        )
        block_setup = [
            dict(base_dim=64 * 2**i, block_depth=3 if i != 2 else l3, heads=heads) for i, heads in enumerate(num_heads)
        ]
        n_blocks = len(block_setup)
        blocks = []
        last_out_dim = 64
        for i, cfg in enumerate(block_setup):
            # print(f"create_block: {cfg}, {last_out_dim}")
            blocks.append(
                _create_halo_block(
                    in_dim=last_out_dim,
                    rv=rv,
                    rb=rb,
                    block_size=block_size,
                    halo_size=halo_size,
                    activation=activation,
                    downsampling=i < n_blocks - 1,
                    qkv_bias=qkv_bias,
                    **cfg,
                )
            )
            last_out_dim = int(cfg["base_dim"] * rb)

        self.blocks = nn.Sequential(*blocks)
        out_blck = [GlobalAvgPool()]
        if embed_dim:
            out_blck += [nn.Linear(last_out_dim, embed_dim), activation]
        self.out_block = nn.Sequential(*out_blck)
        self.embed_dim = embed_dim if embed_dim else last_out_dim
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.num_classes = num_classes

    def forward(self, x):
        b, c, w, h = x.shape
        assert w % 64 == h % 64 == 0, "Image with and height has to be divisible by 64."
        x = self.start_block(x)
        x = self.blocks(x)
        x = self.out_block(x)
        return self.head(x)

    def set_image_res(self, res):
        pass


@register_model
def halonet_h0(pretrained=False, img_size=224, in_chans=3, **kwargs):
    model = HaloNet(**kwargs, in_chans=in_chans, block_size=8, halo_size=3, rv=1.0, rb=0.5, l3=7)
    return model


@register_model
def halonet_h1(pretrained=False, img_size=224, in_chans=3, **kwargs):
    model = HaloNet(**kwargs, in_chans=in_chans, block_size=8, halo_size=3, rv=1.0, rb=1.0, l3=10)
    return model


@register_model
def halonet_h2(pretrained=False, img_size=224, in_chans=3, **kwargs):
    model = HaloNet(**kwargs, in_chans=in_chans, block_size=8, halo_size=3, rv=1.0, rb=1.25, l3=11)
    return model
