from timm.models.vision_transformer import VisionTransformer, Block, PatchEmbed, Attention
from resizing_interface import ResizingInterface


class _MatrixSaveAttn(Attention):
    attn_mat = None
    @classmethod
    def cast(cls, attn: Attention):
        assert isinstance(attn, Attention), f"Can only save attention from Timms attention class"
        attn.__class__ = cls
        assert isinstance(attn, _MatrixSaveAttn)
        return attn

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_mat = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimmViT(VisionTransformer, ResizingInterface):
    """
    Wrapper for *VisionTransformer* from *timm* library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None, class_token=True,
                 no_embed_class=True, pre_norm=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
                 save_attention_maps=False, **kwargs):
        """

        Parameters
        ----------
        img_size : int
            image dimensions -> img_size x img_size
        patch_size : int
            patch_size
        in_chans : int
            number of image channels
        num_classes : int
            number of classes for classification head
        global_pool : str
            type of global pooling for final sequence (default: 'token')
        embed_dim : int
            embedding dimension
        depth : int
            number of transformer layers
        num_heads : int
            number of transformer heads
        mlp_ratio : float
            ratio of feed forward (mlp) hidden dimension to embedding dimension
        qkv_bias : bool
            enable bias for query, key, and value (qkv) embeddings
        init_values : float
            layer scale initial values
        class_token : bool
            use a class token [CLS]
        no_embed_class : bool
            no positional embedding for the class token
        pre_norm : bool
            use pre-norm architecture (norm before the blocks, not after)
        fc_norm : bool
            norm after pool (used when global_pool == 'avrg')
        drop_rate : float
            dropout rate
        attn_drop_rate : float
            dropout rate in the attention module
        drop_path_rate : float
            drop path rate (stochastic depth)
        weight_init : str
            scheme for weight initialization
        embed_layer : nn.Module
            patch embedding layer
        norm_layer : nn.Module
            normalization layer
        act_layer : nn.Module
            activation function
        block_fn : nn.Module
            which block structure to use; for parallel attention layers, ...
        """

        super(TimmViT, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      num_classes=num_classes, global_pool=global_pool, embed_dim=embed_dim, depth=depth,
                                      num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      init_values=init_values, class_token=class_token, no_embed_class=no_embed_class,
                                      pre_norm=pre_norm, fc_norm=fc_norm, drop_rate=drop_rate, proj_drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      weight_init=weight_init, embed_layer=embed_layer, norm_layer=norm_layer,
                                      act_layer=act_layer, block_fn=block_fn)
        self.embed_layer = embed_layer
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pre_norm = pre_norm
        self.class_token = class_token
        self.no_embed_class = no_embed_class
        self.num_classes = num_classes
        self.save_attention_maps = save_attention_maps
        if save_attention_maps:
            self.do_save_attention_maps()

    def do_save_attention_maps(self):
        self.save_attention_maps = True
        for block in self.blocks:
            block.attn = _MatrixSaveAttn.cast(block.attn)

    def attention_maps(self):
        assert self.save_attention_maps, "Have to save attention maps first"
        return [getattr(block.attn, 'attn_mat') for block in self.blocks]
