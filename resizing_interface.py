from abc import ABC
from copy import copy
import torch
from torch import nn


vit_sizes = {
    "na": dict(embed_dim=96, depth=2, num_heads=2),
    "mu": dict(embed_dim=144, depth=6, num_heads=3),
    "Ti": dict(embed_dim=192, depth=12, num_heads=3),
    "S": dict(embed_dim=384, depth=12, num_heads=6),
    "B": dict(embed_dim=768, depth=12, num_heads=12),
    "L": dict(embed_dim=1024, depth=24, num_heads=16),
}


class ResizingInterface(ABC):
    def get_internal_loss(self):
        """Allows a model to add a term to the loss.

        Returns
        -------
        float | torch.Tensor
            internal loss. Default: 0.
        """
        return 0.0

    def set_image_res(self, res):
        """Set a new image resolution -> reset the (learned) patch embedding

        Parameters
        ----------
        res : int
            new image resolution
        """

        if res == self.img_size:
            return

        old_patch_embed_state = copy(self.patch_embed.state_dict())
        self.patch_embed = self.embed_layer(
            img_size=res,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        self.patch_embed.load_state_dict(old_patch_embed_state)

        num_extra_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
        orig_size = int((self.pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(self.patch_embed.num_patches**0.5)
        extra_tokens = self.pos_embed[:, :num_extra_tokens]
        pos_tokens = self.pos_embed[:, num_extra_tokens:]
        # make it shape rest x embed_dim x orig_size x orig_size
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2)
        pos_tokens = nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        # make it shape rest x new_size^2 x embed_dim
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        if num_extra_tokens > 0:
            pos_tokens = torch.cat((extra_tokens, pos_tokens), dim=1)
        self.pos_embed = nn.Parameter(pos_tokens.contiguous())

        self.img_size = res

    def set_num_classes(self, n_classes):
        """Reset the classification head with a new number of classes.

        Parameters
        ----------
        n_classes : int
            new number of classes
        """
        if n_classes == self.num_classes:
            return
        self.head = nn.Linear(self.embed_dim, n_classes) if n_classes > 0 else nn.Identity()
        self.num_classes = n_classes

        # init weight + bias
        # nn.init.zeros_(self.head.weight)
        # nn.init.constant_(self.head.bias, -log(self.num_classes))

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
