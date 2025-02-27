
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import numpy as np

from tractseg.models.segment_anything.modeling.image_encoder import ImageEncoderViT


class Only_Frozen_Sam(torch.nn.Module):
    def __init__(self, n_input_channels=9, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(Only_Frozen_Sam, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

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




    def forward(self, inpt):
        inpt_2 = F.interpolate(inpt, size=(1024,1024), mode='nearest')
        outputs = torch.stack([self.image_encoder(inpt_2[i])])




        for 
        image_embedding = self.image_encoder(inpt_2) # (1,64,64,768)->(1,64,64,768)

        image_embedding_feature = image_embedding.detach().cpu().numpy
        np.save("/home/crq/TractSeg/tractseg/models/image_embedding_feature.npy", image_embedding_feature)


        return 0
    

    
# from torchstat import stat
# device = torch.device("cuda:0") 
# model = UNet_Pytorch_DeepSup_Sam1()
# print(stat(model, (9, 144, 144)))

# model_dict = model.state_dict()
# for k,v in model_dict.items():
#     print(k)


