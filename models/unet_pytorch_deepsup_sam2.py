### 使用FPT之前的代码

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d
############################################
# from mobile_sam import sam_model_registry
from tractseg.models.tiny_vit_sam import TinyViT
#############################################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()

#         self.max_pool = nn.AdaptiveMaxPool2d(1)#NxCx1x1x1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.max_a = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
#             nn.PReLU(),
#             nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
#         )

#         self.avg_a = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
#             nn.PReLU(),
#             nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
#         )

#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         x1 = self.max_pool(x)
#         x2 = self.avg_pool(x)
#         max_out = self.max_a(x1)
#         avg_out = self.avg_a(x2)
#         ma_out = max_out + avg_out
#         out = self.sig(ma_out)
#         out = x.mul(out)
#         return out

class Trans2SAM(nn.Module):
    def __init__(self):
        super(Trans2SAM, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=9, out_channels=40, kernel_size=2, stride=2, padding=0)

        self.conv_2 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=2, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=80, out_channels=160, kernel_size=1, stride=1, padding=0)

    def forward(self, x):        
        x = F.relu(self.conv_1(x)) # 9*144*144 --> 40*72*72
        x = F.interpolate(x, size=(128, 128), mode='bilinear',align_corners=False) #  40*72*72 --> 40*128*128
        x = F.relu(self.conv_2(x))  #  40*128*128 --> 80*64*64
        x = F.relu(self.conv_3(x))  #  80*64*64 --> 160*64*64
        return x


# class FeatureFusion(nn.Module):
#     def __init__(self, channels):
#         super(FeatureFusion, self).__init__()
#         # Voxel-level and channel-level fusion
#         self.Attentionxq = nn.Sequential(
#             ChannelAttention(1024+512+256),
#             nn.Conv2d(1024+512+256, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.Attentionx = nn.Sequential(
#             ChannelAttention(1024+512),
#             nn.Conv2d(1024+512, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.Attentionq = nn.Sequential(
#             ChannelAttention(256),
#             nn.Conv2d(256, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.meta_learner1 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.meta_learner2 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.meta_learner3 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.fuse = nn.Sequential(
#             nn.Conv2d(45, 45, kernel_size=1, bias=False)
#         )

#     def forward(self, pred1, pred2):
#         # Concatenate the predictions
#         xq = torch.cat((pred1, pred2), dim=1) #channels：1024+512+1024
#         xq_features = self.Attentionxq(xq) #channels：1024+512+1024

#         x_features = self.Attentionx(pred1) #1024+512 --> 1024+512+1024
#         q_features = self.Attentionq(pred2) #1024 --> 1024+512+1024

#         # print(xq_features.shape, x_features.shape, q_features.shape)

#         weights1 = self.meta_learner1(xq_features) 
#         weights2 = self.meta_learner2(x_features)
#         weights3 = self.meta_learner3(q_features)
#         xq_fuse = weights1 * xq_features + weights2 * x_features + weights3 * q_features
#         # print(xq_fuse.shape)
#         return xq_fuse ##1024+512+1024

#         #xq_f

# class FeatureFusion(nn.Module):
#     def __init__(self, channels):
#         super(FeatureFusion, self).__init__()
#         # Voxel-level and channel-level fusion
#         self.Attentionxq = nn.Sequential(
#             ChannelAttention(1024+512+256),
#             nn.Conv2d(1024+512+256, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.Attentionx = nn.Sequential(
#             ChannelAttention(1024+512),
#             nn.Conv2d(1024+512, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.Attentionq = nn.Sequential(
#             ChannelAttention(256),
#             nn.Conv2d(256, 1024+512+256, kernel_size=1, bias=True),
#             nn.PReLU()
#         )
#         self.meta_learner1 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.meta_learner2 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.meta_learner3 = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.PReLU(),
#             nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.fuse = nn.Sequential(
#             nn.Conv2d(45, 45, kernel_size=1, bias=False)
#         )

#     def forward(self, pred1, pred2):
#         # Concatenate the predictions
#         xq = torch.cat((pred1, pred2), dim=1) #channels：1024+512+1024
#         xq_features = self.Attentionxq(xq) #channels：1024+512+1024

#         x_features = self.Attentionx(pred1) #1024+512 --> 1024+512+1024
#         q_features = self.Attentionq(pred2) #1024 --> 1024+512+1024

#         # print(xq_features.shape, x_features.shape, q_features.shape)

#         weights1 = self.meta_learner1(xq_features) 
#         weights2 = self.meta_learner2(x_features)
#         weights3 = self.meta_learner3(q_features)
#         xq_fuse = weights1 * xq_features + weights2 * x_features + weights3 * q_features
#         # print(xq_fuse.shape)
#         return xq_fuse ##1024+512+1024

#         #xq_f

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpsamplingBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.deconv(x)
        x = F.relu(x)
        x = self.conv(x)
        return x

class UNet_Pytorch_DeepSup_Sam2(torch.nn.Module):

    # def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=True, dropout=False, upsample="bilinear"):
    def __init__(self, n_input_channels=9, n_classes=7, n_filt=64, batchnorm=True, dropout=False, upsample="bilinear"):
        super(UNet_Pytorch_DeepSup_Sam2, self).__init__()
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

        # self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16 + 256, n_filt * 8)
        # self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16 + n_filt * 16, n_filt * 8)
        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 4 + n_filt * 16, n_filt * 8)
        # self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 4 + n_filt * 16, n_filt * 8)
        # self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        # self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8 , n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8 + n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # 'nearest' a bit faster but results a little worse (~0.4 dice points worse)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        # self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4+ n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4 , n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        # self.expand_4_1 = conv2d(n_filt + n_filt * 2+ n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        
        # 前：
        # self.before_1 = deconv2d(9, 9, kernel_size=2, stride=2) # B*9*144*144 --> 288*288
        # self.before_2 = deconv2d(9, 9, kernel_size=2, stride=2) # 288*288 --> 576*576

        # 后：
        self.after_1 = nn.MaxPool2d(kernel_size=4, stride=4) # B*256*64*64 --> B*256*16*16
        # self.after_1 = conv2d(256, 256, 4, 4) # B*256*64*64 --> B*256*16*16 先减小尺寸
        # self.after_2 = conv2d(256, 1024, 1, 1) # B*256*16*16 --> B*1024*16*16 再增加维度
        # F.interpolate((18, 18)) B*1024*16*16-->B*1024*18*18

        # self.fusion_module =  FeatureFusion(1024+512+256)
        # self.fusion_module =  FeatureFusion(1024+512+1024)

        self.trans2sam = Trans2SAM()
        
        model_weight_path = "/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_4.pt"
        
        # model_type = "vit_t"
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # mobile_sam = sam_model_registry[model_type](
        #     checkpoint=None, 
        #     # checkpoint=model_weight_path, 
        #     # image_size = 144,
        #     # in_chans = 9,
        #     # prompt_embed_dim = 1024,
        #     # vit_patch_size = 8,
        #     )
        ### expected size of image_embedding: B*prompt_embed_dim*H*W
        ### 56*1024*18*18 
        # mobile_sam.to(device=device)   

        # self.sam_image_encoder = mobile_sam.image_encoder

        # for name, param in self.sam_image_encoder.named_parameters():
        #     param.requires_grad = True
        self.sam_image_encoder=TinyViT(img_size=64, in_chans=160, num_classes=0,
            embed_dims=[160, 320],
            depths=[2, 2],
            num_heads=[5, 10],
            window_sizes=[14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )

        with open(model_weight_path, "rb") as f:
            state_dict = torch.load(f)
            self.sam_image_encoder.load_state_dict(state_dict, strict=False)

        for name, param in self.sam_image_encoder.named_parameters():
            param.requires_grad = True
            # param.requires_grad = False
            
           
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################
        self.ffm = FFM()


    def forward(self, inpt):
        # inpt.shape: torch.Size([56, 9, 144, 144])

        ###################################################################################
        ###################################################################################
        ###################################################################################
        # 输入SAM前：先将输入的input peaks（B*9*144*144）经过卷积等操作变化到SAM能处理的shape
        # SAM输出后：将SAM的输出经过卷积等操作变化到能拼接到UNet网络中间

        # # 前:
        # inpt2 = self.before_1(inpt) # B*9*144*144 --> 288*288
        # inpt2 = self.before_2(inpt2) #288*288 --> 576*576
        # inpt2 = F.interpolate(inpt2, (1024,1024), mode='bilinear', align_corners=False) # 576*576 --> 1024*1024
        # # inpt2: B*9*1024*1024 
        # # print(inpt2.shape)
        # image_embedding = self.sam_image_encoder(inpt2); ###### 输出：(B*256*64*64)
        # # print("image_embedding.shape:", image_embedding.shape)

        # # 后
        # image_embedding = self.after_1(image_embedding)  # B*256*64*64 --> B*256*16*16
        # image_embedding = self.after_2(image_embedding)  # B*256*64*64 --> B*1024*16*16
        # image_embedding = F.interpolate(image_embedding, size=(18,18), mode='bilinear', align_corners=False) 
        # # B*1024*16*16上采样成B*1024*18*18

        # 后 直接采用256

        image_embedding_before = self.trans2sam(inpt) # B*160*64*64
        image_embedding = self.sam_image_encoder(image_embedding_before) # # B*256*64*64


        image_embedding = self.after_1(image_embedding)  # B*256*64*64 --> B*256*16*16
        image_embedding = F.interpolate(image_embedding, size=(18,18), mode='bilinear', align_corners=False) 


        ################################################################################### 

        contr_1_1 = self.contr_1_1(inpt)        # 144*144*9 --> 144*144*64
        contr_1_2 = self.contr_1_2(contr_1_1)   # 144*144*64 --> 144*144*64
        pool_1 = self.pool_1(contr_1_2)         # 144*144*64 --> 72*72*64

        contr_2_1 = self.contr_2_1(pool_1)      # 72*72*64 -->72*72*128
        contr_2_2 = self.contr_2_2(contr_2_1)   # 72*72*128 --> 72*72*128
        pool_2 = self.pool_2(contr_2_2)         # 72*72*128 --> 36*36*128

        contr_3_1 = self.contr_3_1(pool_2)      # 36*36*128 --> 36*36*256
        contr_3_2 = self.contr_3_2(contr_3_1)   # 36*36*256 --> 36*36*256
        pool_3 = self.pool_3(contr_3_2)         # 36*36*256 --> 18*18*256

        contr_4_1 = self.contr_4_1(pool_3)      # 18*18*256 -->18*18*512
        contr_4_2 = self.contr_4_2(contr_4_1)   # 18*18*512 -->18*18*512
        pool_4 = self.pool_4(contr_4_2)         # 18*18*512 -->9*9*512

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)       # 9*9*512 --> 9*9*512

        encode_1 = self.encode_1(pool_4)        # 9*9*512 --> 9*9*1024
        encode_2 = self.encode_2(encode_1)      # 9*9*1024 --> 9*9*1024
        deconv_1 = self.deconv_1(encode_2)      # 9*9*1024 --> 18*18*1024

        # concat1 = torch.cat([deconv_1, contr_4_2], 1) # 18*18*(1024+512)
        # concat1 = torch.cat([deconv_1, contr_4_2, image_embedding], 1) # 18*18*(1024+512+1024)
        concat1 = torch.cat([deconv_1, contr_4_2], 1) # 18*18*(1024+512+1024)
        concat1 = self.ffm(image_embedding, concat1)
        
        
        # concat1 = torch.cat([deconv_1, contr_4_2], 1) # 18*18*(1024+512)
        # print(concat1.shape)
        # print(deconv_1.shape)
        # print(contr_4_2.shape)

        # concat1 = self.fusion_module(concat1, image_embedding)
        # print(concat1.shape)
        # print(concat1.shape)
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################
        expand_1_1 = self.expand_1_1(concat1)   # 18*18*(1024+512) --> 18*18*512
        expand_1_2 = self.expand_1_2(expand_1_1) # 18*18*512 --> 18*18*512
        deconv_2 = self.deconv_2(expand_1_2)     # 18*18*512 -->36*36*512

        concat2 = torch.cat([deconv_2, contr_3_2], 1) # 36*36*(512+256)
        # concat2 = torch.cat([deconv_2, contr_3_2, image_embedding_2], 1) # 36*36*(512+256)
        expand_2_1 = self.expand_2_1(concat2)     # 36*36*(512+256) --> # 36*36*256
        expand_2_2 = self.expand_2_2(expand_2_1)  # 36*36*256 --> 36*36*256
        deconv_3 = self.deconv_3(expand_2_2)      # 36*36*256 --> 72*72*256

        # Deep Supervision
        output_2 = self.output_2(concat2)         # 36*36*(512+256) --> 36*36*classes
        output_2_up = self.output_2_up(output_2)  # 36*36*classes --> 72*72*classes

        concat3 = torch.cat([deconv_3, contr_2_2], 1) # 72*72*(256+128)
        # concat3 = torch.cat([deconv_3, contr_2_2, image_embedding_3], 1) # 72*72*(256+128)
        expand_3_1 = self.expand_3_1(concat3)     # 72*72*(256+128) --> # 72*72*128
        expand_3_2 = self.expand_3_2(expand_3_1)  # 72*72*128 -->  72*72*128
        deconv_4 = self.deconv_4(expand_3_2)      # 72*72*128 --> 144*144*128

        # Deep Supervision
        # output_3(concat3)  # 72*72*(256+128) --> 72*72*classes
        output_3 = output_2_up + self.output_3(concat3) # 72*72*classes + 72*72*classes = 72*72*classes
        output_3_up = self.output_3_up(output_3)    # 72*72*classes --> 144*144*classes

        concat4 = torch.cat([deconv_4, contr_1_2], 1) # 144*144*(128+64)
        # concat4 = torch.cat([deconv_4, contr_1_2, image_embedding_4], 1) # 144*144*(128+64)
        expand_4_1 = self.expand_4_1(concat4)       # 144*144*(128+64) --> 144*144*64
        expand_4_2 = self.expand_4_2(expand_4_1)    # 144*144*64 --> 144*144*64

        conv_5 = self.conv_5(expand_4_2)            # 144*144*64 --> 144*144*classes

        final = output_3_up + conv_5                # 144*144*classes + 144*144*classes = 144*144*classes

        return final                                # 144*144*classes



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CrossAttention(nn.Module):
    def __init__(self, query_channels, key_channels, value_channels):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Conv2d(query_channels, query_channels, kernel_size=1, bias=False)
        self.W_k = nn.Conv2d(key_channels, query_channels, kernel_size=1, bias=False)
        self.W_v = nn.Conv2d(value_channels, query_channels, kernel_size=1, bias=False)  # 修改输出通道数为 query_channels

    def forward(self, query, key, value):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        b, c, h, w = q.size()
        q = q.view(b, c, -1).permute(0, 2, 1)  # [b, hw, c]
        k = k.view(b, c, -1)  # [b, c, hw]
        v = v.view(b, c, -1).permute(0, 2, 1)  # [b, hw, c]

        attn_score = torch.softmax(torch.matmul(q, k) / torch.sqrt(torch.tensor(c, dtype=torch.float32)), dim=-1)
        out = torch.matmul(attn_score, v).permute(0, 2, 1).view(b, c, h, w)
        return out


class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()
        # 调整 SAM 特征图通道数到 1536 以匹配 CNN 特征图
        self.sam_channel_adjust = nn.Conv2d(256, 1536, kernel_size=1, bias=False)

        # 通道注意力机制
        self.sam_channel_attn = ChannelAttention(256)
        self.cnn_channel_attn = ChannelAttention(1536)
        self.concat_channel_attn = ChannelAttention(3072)

        # 交叉注意力机制
        self.cross_attn = CrossAttention(3072, 256, 1536)

        # 最终卷积层，输出通道数为 1792
        self.final_conv = nn.Conv2d(3072, 1792, kernel_size=3, padding=1, bias=False)

    def forward(self, sam_feature, cnn_feature):
        # 通道注意力
        sam_attn = self.sam_channel_attn(sam_feature)
        cnn_attn = self.cnn_channel_attn(cnn_feature)

        # 调整 SAM 特征图的通道数
        sam_adjusted = self.sam_channel_adjust(sam_attn)

        # 特征拼接
        concat_feature = torch.cat([sam_adjusted, cnn_attn], dim=1)

        # 拼接后的通道注意力
        concat_attn = self.concat_channel_attn(concat_feature)

        # 交叉注意力
        cross_attn_out = self.cross_attn(concat_attn, sam_attn, cnn_attn)
        print("cross_attn_out shape:", cross_attn_out.shape)  # 添加打印语句，查看形状

        # 最终卷积
        final_feature = self.final_conv(cross_attn_out)

        return final_feature
    def __init__(self):
        super(FFM, self).__init__()
        # 调整 SAM 特征图通道数到 1536 以匹配 CNN 特征图
        self.sam_channel_adjust = nn.Conv2d(256, 1536, kernel_size=1, bias=False)

        # 通道注意力机制
        self.sam_channel_attn = ChannelAttention(256)
        self.cnn_channel_attn = ChannelAttention(1536)
        self.concat_channel_attn = ChannelAttention(3072)

        # 交叉注意力机制
        self.cross_attn = CrossAttention(3072, 256, 1536)

        # 最终卷积层，输出通道数为 1792
        self.final_conv = nn.Conv2d(3072, 1792, kernel_size=3, padding=1, bias=False)

    def forward(self, sam_feature, cnn_feature):
        # 通道注意力
        sam_attn = self.sam_channel_attn(sam_feature)
        cnn_attn = self.cnn_channel_attn(cnn_feature)

        # 调整 SAM 特征图的通道数
        sam_adjusted = self.sam_channel_adjust(sam_attn)

        # 特征拼接
        concat_feature = torch.cat([sam_adjusted, cnn_attn], dim=1)

        # 拼接后的通道注意力
        concat_attn = self.concat_channel_attn(concat_feature)

        # 交叉注意力
        cross_attn_out = self.cross_attn(concat_attn, sam_attn, cnn_attn)

        # 最终卷积
        final_feature = self.final_conv(cross_attn_out)

        return final_feature
    
def important_token_selection(key_layer, value_layer, attention_probs, token_ratio):
    N = attention_probs.shape[-1]
    num_patch = int(N * token_ratio)
    
    highlights = attention_probs.mean(dim=-2).topk(num_patch, dim=-1).indices

    highlights = highlights.unsqueeze(-1).repeat(1, 1, 1, key_layer.shape[-1])
    k_highlights = torch.gather(key_layer, dim=-2, index=highlights)
    v_highlights = torch.gather(value_layer, dim=-2, index=highlights)
