
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import math
# import SimpleITK as sitk
from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d
import torch.cuda.comm as comm
import torch.nn.functional as F
import os, time
import functools
from torch.nn import Softmax


# from collections import OrderedDict
# from torchsummary import summary  # pip install torchsummary

def conv1x1(in_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, bias=False)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Sequential):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        # print(m_batchsize, _, height, width)
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height
                                                                                                     ,height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W)     # + x


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # out = F.avg_pool2d(out, 2)
        return out


class DenseCCABlock(nn.Sequential):
    def __init__(self, in_channels):
        super(DenseCCABlock, self).__init__()
        self.ccablock = CrissCrossAttention(in_channels)
        self.transition_1 = Transition( 2 *in_channels, in_channels)
        self.transition_2 = Transition( 3 *in_channels, in_channels)

    def forward(self, x):
        ccablock_1 = self.ccablock(x)
        concat1 = torch.cat([ccablock_1, x], 1)
        transition_1 = self.transition_1(concat1)

        ccablock_2 = self.ccablock(transition_1)
        concat2 = torch.cat([(torch.cat([ccablock_2, ccablock_1], 1)), x], 1)
        transition_2 = self.transition_2(concat2)

        return transition_2

class UNet_FAM(torch.nn.Module):
    def __init__(self, n_input_channels, n_classes=1, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(UNet_FAM, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

        self.conv1x1 = conv1x1(1024)

        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.contr_1_1 = conv2d(n_input_channels, n_filt)
        self.contr_1_2 = conv2d(n_filt, n_filt)
        self.pool_1 = nn.MaxPool2d((2, 2))
        #Adaptive_avg_pool
        #self.GAP_1 = nn.functional.adaptive_avg_pool2d(input,(144,144))
        self.GAP_1 = nn.AdaptiveMaxPool2d((72,72))
        self.contr_2_1 = conv2d(n_filt, n_filt * 2)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2)
        self.pool_2 = nn.MaxPool2d((2, 2))

        #self.GAP_2 = nn.functional.adaptive_avg_pool2d(input,(72,72))
        self.GAP_2 = nn.AdaptiveMaxPool2d((36, 36))
        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16)
        self.atte5_1 = DenseCCABlock(in_channels=n_filt * 16)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 64)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)
        self.deconv_1_2 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=4, stride=4)
        # self.deconv_1 = nn.Upsample(scale_factor=2)  # does only upscale width and height; Similar results to deconv2d

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 32)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)
        # self.deconv_2 = nn.Upsample(scale_factor=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 16, stride=1)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)
        # self.deconv_3 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_2 = nn.Conv2d(n_filt * 4 + n_filt * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # 'nearest' a bit faster but results a little worse (~0.4 dice points worse)
        self.output_2_up = nn.Upsample(scale_factor=2, mode=upsample)

        self.expand_3_1 = conv2d(n_filt * 4, n_filt * 2, stride=1)
        #self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 8, stride=1)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)
        # self.deconv_4 = nn.Upsample(scale_factor=2)

        # Deep Supervision
        self.output_3 = nn.Conv2d(n_filt * 2 + n_filt * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height

        self.expand_4_1 = conv2d(n_filt * 2, n_filt, stride=1)
        #self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1)

        # no activation function, because is in LossFunction (...WithLogits)
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.ps = nn.PixelShuffle(2)
        self.ps_2 = nn.PixelShuffle(2)
        self.ps_3 = nn.PixelShuffle(2)
        self.ps_4 = nn.PixelShuffle(2)
        self.conv_output_1 = conv2d(4096, 72)
        self.conv_output_2 = conv2d(2048, 72)

        self.conv_gap_1_1 = conv2d(64,256)
        self.conv_gap_1_2 = conv2d(256,512)
        self.conv_gap_2_1 = conv2d(128,512)
        self.conv_gap_2_2 = conv2d(512,1024)

        self.conv_output_3 = conv2d(1024, 72)
        self.conv_output_4 = conv2d(512, 72)

    def forward(self, inpt):
        contr_1_1 = self.contr_1_1(inpt)
        contr_1_2 = self.contr_1_2(contr_1_1)
        pool_1 = self.pool_1(contr_1_2)
        gap_1 = self.GAP_1(contr_1_2)
        #print(gap_1.shape)

        contr_2_1 = self.contr_2_1(pool_1)
        contr_2_2 = self.contr_2_2(contr_2_1)
        pool_2 = self.pool_2(contr_2_2)
        gap_2 = self.GAP_2(contr_2_2)
        #print(gap_2.shape)

        contr_3_1 = self.contr_3_1(pool_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1(pool_3)
        contr_4_2 = self.contr_4_2(contr_4_1)
        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1(pool_4)
        atte5 = self.atte5_1(encode_1)
        encode_2 = self.encode_2(atte5)
        output_1 = self.conv_output_1(encode_2)
        # atte_w = self.conv1x1(atte5)
        # atte_w = self.bn(atte_w)
        # atte_w = self.sigmoid(atte_w)
        # print(atte4.shape, atte3.shape, atte2.shape, atte1.shape)
        # deconv_1 = self.deconv_1(atte5)
        deconv_1 = self.ps(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)
        output_2 = self.conv_output_2(expand_1_2)
        # deconv_2 = self.deconv_2(expand_1_2)
        deconv_2 = self.ps_2(expand_1_2)


        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        output_3 = self.conv_output_3(expand_2_2)

        Downscaling_1 = self.conv_gap_2_1(gap_2)
        mul_conv_2 = self.conv_gap_2_2(Downscaling_1)
        mul_2 = torch.mul(mul_conv_2,expand_2_2)
        add_2 = torch.add(mul_2,expand_2_2)

        # deconv_3 = self.deconv_3(expand_2_2)
        deconv_3 = self.ps_3(add_2)

        #Deep Supervision
        output_2 = self.output_2(concat2)
        output_2_up = self.output_2_up(output_2)

        #concat3 = torch.cat([deconv_3, contr_2_2], 1)

        expand_3_1 = self.expand_3_1(deconv_3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        output_4 = self.conv_output_4(expand_3_2)
        # deconv_4 = self.deconv_4(expand_3_2)

        Downscaling_2 = self.conv_gap_1_1(gap_1)
        mul_conv_1 = self.conv_gap_1_2(Downscaling_2)
        mul_1 = torch.mul(mul_conv_1, expand_3_2)
        add_1 = torch.add(mul_1, expand_3_2)

        deconv_4 = self.ps_4(add_1)

        # #Deep Supervision
        # output_3 = output_2_up + self.output_3(expand_3_1)
        # #output_3 = output_2_up + self.output_3(concat3)
        # output_3_up = self.output_3_up(output_3)

        #concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(deconv_4)
        expand_4_2 = self.expand_4_2(expand_4_1)


        conv_5 = self.conv_5(expand_4_2)

        final = conv_5
        #final = conv_5
        # output_1_1 = F.interpolate(output_1, scale_factor=16, mode='bilinear')
        # output_2_1 = F.interpolate(output_2, scale_factor=8, mode='bilinear')
        # output_3_1 = F.interpolate(output_3, scale_factor=4, mode='bilinear')
        # output_4_1 = F.interpolate(output_4, scale_factor=2, mode='bilinear')
        #
        # return [output_1_1, output_2_1, output_3_1, output_4_1, final]
        return final
        print('sucess')

# input = torch.rand([1, 9, 144, 144]).cuda()
# model = UNet_FAM(9).cuda()
# out = model(input)
