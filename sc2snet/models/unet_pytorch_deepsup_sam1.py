
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from typing import Optional, Tuple, Type

from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d
from tractseg.models.segment_anything.modeling.image_encoder import ImageEncoderViT
from tractseg.models.segment_anything.modeling.common import LayerNorm2d, MLPBlock


class UNet_Pytorch_DeepSup_Sam1(torch.nn.Module):
    def __init__(self, n_input_channels=9, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(UNet_Pytorch_DeepSup_Sam1, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        # self.deconv_1 = nn.Upsample(scale_factor=2)  # does only upscale width and height; Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16 + 768, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # 'nearest' a bit faster but results a little worse (~0.4 dice points worse)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.patch_embed = PatchEmbed()

        self.image_encoder_weight_path = '/home/crq/MedSAM/checkpoint/medsam_vit_b.pth'
        # self.image_encoder_weight_path = '/home/crq/segment-anything-main/checkpoints/sam_vit_b_01ec64.pth'

        self.image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )



        with open(self.image_encoder_weight_path, "rb") as f:
            state_dict = torch.load(f)
            modified_state_dict = {
                k.replace('image_encoder.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('image_encoder.')
                }
            modified_state_dict = {
                k: v 
                for k, v in modified_state_dict.items()
                if 'patch_embed' not in k and 'neck' not in k
            }
            
            self.image_encoder.load_state_dict(modified_state_dict)


        for _, param in self.image_encoder.named_parameters():
            param.requires_grad = True
        
        # for k, v in modified_state_dict.items():
        #     print(k, v.shape)

        self.neck = nn.Sequential(
            nn.Conv2d(
                768,
                768,
                kernel_size=4,
                stride=4,
                bias=False,
            ),
            LayerNorm2d(768),
            nn.Conv2d(
                768,
                768,
                kernel_size=1,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(768),
        )



    def forward(self, inpt):

        inpt_2 = F.interpolate(inpt, size=(1024,1024), mode='nearest')
        inpt_2 = self.patch_embed(inpt_2)
        image_embedding = self.image_encoder(inpt_2) # (1,64,64,768)->(1,64,64,768)
        image_embedding = self.neck(image_embedding.permute(0, 3, 1, 2)) # ->(1,768,18,18)

        
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        encode_2 = self.encode_2(encode_1)
        deconv_1 = self.deconv_1(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)

        concat1 = torch.cat([concat1, image_embedding], 1)


        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        deconv_2 = self.deconv_2(expand_1_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        deconv_3 = self.deconv_3(expand_2_2)

        # Deep Supervision
        output_2 = self.output_2(concat2)
        output_2_up = self.output_2_up(output_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        deconv_4 = self.deconv_4(expand_3_2)

        # Deep Supervision
        output_3 = output_2_up + self.output_3(concat3)
        output_3_up = self.output_3_up(output_3)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        conv_5 = self.conv_5(expand_4_2)

        final = output_3_up + conv_5

        return final
    
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 9,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
    
# from torchstat import stat
# device = torch.device("cuda:0") 
# model = UNet_Pytorch_DeepSup_Sam1()
# print(stat(model, (9, 144, 144)))

# model_dict = model.state_dict()
# for k,v in model_dict.items():
#     print(k)