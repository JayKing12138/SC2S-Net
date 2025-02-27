# from torch import nn
# from typing import Tuple, Union
# from unetr_pp.network_architecture.neural_network import SegmentationNetwork
# from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
# from unetr_pp.network_architecture.lung.model_components import UnetrPPEncoder, UnetrUpBlock


# class UNETR_PP(SegmentationNetwork):
#     """
#     UNETR++ based on: "Shaker et al.,
#     UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#     """
#     def __init__(
#             self,
#             n_input_channels: int,
#             # in_channels: int,
#             out_channels: int = 1,
#             feature_size: int = 16,
#             hidden_size: int = 256,
#             num_heads: int = 4,
#             pos_embed: str = "perceptron",
#             norm_name: Union[Tuple, str] = "instance",
#             dropout_rate: float = 0.0,
#             depths=None,
#             dims=[32, 64, 128, 256],
#             # dims=None,
#             conv_op=nn.Conv3d,
#             do_ds=False,
#             # do_ds=True,
#             n_classes=True,
#             n_filt=True,
#             batchnorm=True,
#             dropout=True,
#             upsample=True

#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimensions of  the last encoder.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             dropout_rate: faction of the input units to drop.
#             depths: number of blocks for each stage.
#             dims: number of channel maps for the stages.
#             conv_op: type of convolution operation.
#             do_ds: use deep supervision to compute the loss.
#         """

#         super().__init__()
#         if depths is None:
#             depths = [3, 3, 3, 3]
#         self.do_ds = do_ds
#         self.conv_op = conv_op
#         self.num_classes = out_channels
#         if not (0 <= dropout_rate <= 1):
#             raise AssertionError("dropout_rate should be between 0 and 1.")

#         if pos_embed not in ["conv", "perceptron"]:
#             raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

#         self.feat_size = (4, 6, 6,)
#         self.hidden_size = hidden_size

#         self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

#         self.encoder1 = UnetResBlock(
#             spatial_dims=3,
#             in_channels=n_input_channels,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#         )
#         self.decoder5 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 16,
#             out_channels=feature_size * 8,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=8*12*12,
#         )
#         self.decoder4 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 8,
#             out_channels=feature_size * 4,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=16*24*24,
#         )
#         self.decoder3 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 4,
#             out_channels=feature_size * 2,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=32*48*48,
#         )
#         self.decoder2 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 2,
#             out_channels=feature_size,
#             kernel_size=3,
#             upsample_kernel_size=(1, 4, 4),
#             norm_name=norm_name,
#             out_size=32*192*192,
#             conv_decoder=True,
#         )
#         self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
#         if self.do_ds:
#             self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
#             self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x

#     def forward(self, x_in):
#         #print("#####input_shape:", x_in.shape)
#         x_in = x_in.unsqueeze(1)
#         x_output, hidden_states = self.unetr_pp_encoder(x_in)

#         convBlock = self.encoder1(x_in)

#         # Four encoders
#         enc1 = hidden_states[0]
#         #print("ENC1:",enc1.shape)
#         enc2 = hidden_states[1]
#         #print("ENC2:",enc2.shape)
#         enc3 = hidden_states[2]
#         #print("ENC3:",enc3.shape)
#         enc4 = hidden_states[3]
#         #print("ENC4:",enc4.shape)

#         # Four decoders
#         dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
#         dec3 = self.decoder5(dec4, enc3)
#         dec2 = self.decoder4(dec3, enc2)
#         dec1 = self.decoder3(dec2, enc1)

#         out = self.decoder2(dec1, convBlock)
#         if self.do_ds:
#             logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
#         else:
#             logits = self.out1(out)

#         return logits

import torch
import torch.nn as nn
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.lung.transformerblock import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock

einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=3,  # 修改 in_channels 为 3，与原网络一致
                 dropout=0.0, transformer_dropout_rate=0.15, **kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size=out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class UNETR_PP(nn.Module):
    def __init__(self, n_input_channels=9, n_classes=1, **kwargs):
        super().__init__()
        self.encoder = UnetrPPEncoder(in_channels=n_input_channels, **kwargs)
        # 这里需要根据具体的UNETR++结构调整解码器部分，确保输出通道数为 n_classes
        self.decoder = UnetrUpBlock(spatial_dims=3, in_channels=256, out_channels=n_classes, kernel_size=3,
                                    upsample_kernel_size=2, norm_name=("group", {"num_groups": 8}), conv_decoder=True)
        self.final_conv = nn.Conv3d(256, n_classes, kernel_size=1)  # 调整为最终输出通道数为 n_classes

    def forward(self, x):
        x, hidden_states = self.encoder(x)
        # 这里需要根据具体的UNETR++结构调整解码器的输入和连接方式
        out = self.decoder(x, hidden_states[-1])
        out = self.final_conv(out)
        return out