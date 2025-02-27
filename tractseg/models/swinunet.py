# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False,
        n_classes=True,
        n_filt=True,
        batchnorm=True,
        dropout=True,
        upsample=True,
        n_input_channels=9):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.swin_unet = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads= [3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        
        self.conv9to3 = nn.Conv2d(in_channels=9, out_channels=3,kernel_size=1)


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear',align_corners=False)
        x = self.conv9to3(x)
        logits = self.swin_unet(x)
        
        # 调整输出尺寸
        target_size = (144, 144)  # 假设目标尺寸是 144x144
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        
        return logits

    # 其他代码保持不变