from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# 自定义 2D 卷积层
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding)


# 自定义反卷积层
def deconv2d(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride)


# Transformer 编码器中的多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        return output


# Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x


# Transformer 编码器
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# TransUNet 模型
class TransUNet(nn.Module):
    def __init__(self, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear",
                 num_layers=4, num_heads=4, mlp_dim=256):
        super(TransUNet, self).__init__()
        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.use_dropout = dropout

        # CNN 编码器
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

        self.dropout_layer = nn.Dropout(p=0.4)

        # 展平特征图用于 Transformer 输入
        self.flatten = nn.Flatten(start_dim=2)

        # 先不初始化线性层，在第一次前向传播时根据输入确定输入维度
        self.linear_proj = None

        # Transformer 编码器
        self.transformer_encoder = TransformerEncoder(num_layers, n_filt * 8, num_heads, mlp_dim, dropout=0.1)

        # 解码器
        self.deconv_1 = deconv2d(n_filt * 8, n_filt * 8)
        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 8, n_filt * 8)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8)

        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 4)
        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 4, n_filt * 4)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4)

        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 2)
        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 2, n_filt * 2)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2)

        self.deconv_4 = deconv2d(n_filt * 2, n_filt)
        self.expand_4_1 = conv2d(n_filt + n_filt, n_filt)
        self.expand_4_2 = conv2d(n_filt, n_filt)

        # 输出层
        self.conv_5 = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, inpt):
        # CNN 编码器
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
            pool_4 = self.dropout_layer(pool_4)

        # 展平特征图
        flat_features = self.flatten(pool_4)

        # 如果线性层还未初始化，根据当前输入确定输入维度并初始化
        if self.linear_proj is None:
            input_dim = flat_features.size(1)
            self.linear_proj = nn.Linear(input_dim, pool_4.size(1)).to(inpt.device)

        proj_features = self.linear_proj(flat_features.permute(0, 2, 1))

        # Transformer 编码器
        trans_features = self.transformer_encoder(proj_features)
        trans_features = trans_features.permute(0, 2, 1).view(-1, pool_4.size(1), pool_4.size(2), pool_4.size(3))

        # 解码器
        deconv_1 = self.deconv_1(trans_features)
        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1(concat1)
        expand_1_2 = self.expand_1_2(expand_1_1)

        deconv_2 = self.deconv_2(expand_1_2)
        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1(concat2)
        expand_2_2 = self.expand_2_2(expand_2_1)

        deconv_3 = self.deconv_3(expand_2_2)
        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1(concat3)
        expand_3_2 = self.expand_3_2(expand_3_1)

        deconv_4 = self.deconv_4(expand_3_2)
        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1(concat4)
        expand_4_2 = self.expand_4_2(expand_4_1)

        # 输出层
        conv_5 = self.conv_5(expand_4_2)

        return conv_5