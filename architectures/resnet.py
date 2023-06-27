from timm.models import register_model
from timm.models.resnet import ResNet as ResNetTimm, BasicBlock, Bottleneck
from resizing_interface import ResizingInterface


class ResNet(ResNetTimm, ResizingInterface):
    def __init__(self, *args, global_pool='avg', **kwargs):
        admissible_kwargs = ["block", "layers", "num_classes", "in_chans", "output_stride", "cardinality", "base_width",
                             "stem_width", "stem_type", "replace_stem_pool", "block_reduce_first", "down_kernel_size",
                             "avg_down", "act_layer", "norm_layer", "aa_layer", "drop_rate", "drop_path_rate",
                             "drop_block_rate", "zero_init_last", "block_args"]
        for key in list(kwargs.keys()):
            if key not in admissible_kwargs:
                kwargs.pop(key)
        super().__init__(*args, global_pool=global_pool, **kwargs)
        self.global_pool_str = global_pool

    def set_image_res(self, res):
        # resizing not needed for CNNs with pooling
        return

    def set_num_classes(self, n_classes):
        if self.num_classes == n_classes:
            return
        self.reset_classifier(num_classes=n_classes, global_pool=self.global_pool_str)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return ResNet(**model_args, **kwargs)

@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return ResNet(**model_args, **kwargs)

@register_model
def resnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return ResNet(**model_args, **kwargs)

@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return ResNet(**model_args, **kwargs)

@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return ResNet( **model_args, **kwargs)

@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return ResNet(**model_args, **kwargs)