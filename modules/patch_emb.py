"""
This code is build on timm package.
"""
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed
from timm.layers.format import Format, nchw_to


from modules.patch_n_pack import patch_n_pack

_logger = logging.getLogger(__name__)


class PyramidPatchEmbed(PatchEmbed):
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            patch_level: int = 2,
            *args, **kwargs
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
            strict_img_size=strict_img_size,
            dynamic_img_pad=dynamic_img_pad,
            *args, **kwargs
        )
        if patch_level == 2:
            self.proj1 = nn.Conv2d(in_chans * 4, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
            self.pyramid_projs = {
                '0': self.proj1,
            }
        elif patch_level == 3:
            self.proj1 = nn.Conv2d(in_chans * 4, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
            self.proj2 = nn.Conv2d(in_chans * 2, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        
            self.pyramid_projs = {
                '0': self.proj1,
                '1': self.proj2,
            }
        self.patch_level = patch_level

    def forward(self, x, token_length_list,  x_duration=None, channel_list=None, training=True) -> torch.Tensor:
        bz = len(x)
        x_token_list = []
        if training:
            x_tmp = x
            for i in range(len(token_length_list)):
                x = torch.stack(x_tmp[round((bz*x_duration[i]).item()):round((bz*x_duration[i+1]).item())], dim=0)
                x = self.pyramid_projs.get(str(i), self.proj)(x)
                if self.flatten:
                    x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
                elif self.output_fmt != Format.NCHW:
                    x = nchw_to(x, self.output_fmt)
                x = self.norm(x)
                x_token_list.append(x)

            del x_tmp, x
            
            x, attention_mask = patch_n_pack(x_token_list, token_length_list)
            return x, attention_mask

        else:
            channel_idx = channel_list.index(x.shape[1])
            x = self.pyramid_projs.get(str(channel_idx), self.proj)(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            elif self.output_fmt != Format.NCHW:
                x = nchw_to(x, self.output_fmt)
            x = self.norm(x)
            attention_mask = torch.ones((bz, x.shape[1], x.shape[1]), dtype=x[0].dtype, device=x[0].device)
            return x, attention_mask