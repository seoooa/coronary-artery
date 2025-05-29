# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F

import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from monai.inferers import sliding_window_inference


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class ModifiedUnetrUpBlock(nn.Module):
    def __init__(
        self, spatial_dims, in_channels, out_channels, norm_name, res_block, label_nc
    ):
        super().__init__()
        # First apply regular convolution operations
        self.transp_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        # SPADE normalization
        self.spade = SPADE(out_channels, label_nc)

    def forward(self, x, skip, segmap=None):
        # Upsampling
        up = self.transp_conv(x)
        if segmap is not None:
            up = self.spade(up, segmap)
        # Concatenate with skip connection
        out = torch.cat((up, skip), dim=1)
        # Apply convolution
        # Apply SPADE if segmap is provided

        out = self.conv_block(out)

        return out


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        label_nc: int = 1,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(
            img_d // p_d for img_d, p_d in zip(img_size, self.patch_size)
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            norm_name=norm_name,
            res_block=res_block,
            label_nc=label_nc,
        )
        self.decoder4 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            norm_name=norm_name,
            res_block=res_block,
            label_nc=label_nc,
        )
        self.decoder3 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            norm_name=norm_name,
            res_block=res_block,
            label_nc=label_nc,
        )
        self.decoder2 = ModifiedUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            norm_name=norm_name,
            res_block=res_block,
            label_nc=label_nc,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )
        self.proj_axes = (0, spatial_dims + 1) + tuple(
            d + 1 for d in range(spatial_dims)
        )
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, segmap):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4, segmap)
        dec2 = self.decoder4(dec3, enc3, segmap)
        dec1 = self.decoder3(dec2, enc2, segmap)
        out = self.decoder2(dec1, enc1, None)
        return self.out(out)


class SPADEUNETR(nn.Module):
    """
    UNETR variant with SPADE normalization in both encoder and decoder paths.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize UNETR backbone
        self.unetr = UNETR(*args, **kwargs)
        self.label_nc = kwargs.get("label_nc", 1)

    def forward(self, x_in, segmap=None):
        # Regular UNETR forward pass
        return self.unetr(x_in, segmap)

    def inference(
        self, x_in, segmap=None, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5
    ):
        """
        Perform sliding window inference with proper handling of segmap
        """
        self.eval()
        with torch.no_grad():
            if segmap is not None:
                # Combine input and segmap for sliding window
                combined_input = torch.cat((x_in, segmap), dim=1)

                def _inner_predict(inputs):
                    # Split combined input back into x_in and segmap
                    x = inputs[:, : x_in.shape[1], ...]
                    seg = inputs[:, x_in.shape[1] :, ...]
                    return self.forward(x, seg)

                return sliding_window_inference(
                    combined_input,
                    roi_size,
                    sw_batch_size,
                    _inner_predict,
                    overlap=overlap,
                )
            else:
                return sliding_window_inference(
                    x_in,
                    roi_size,
                    sw_batch_size,
                    lambda x: self.forward(x, None),
                    overlap=overlap,
                )


# Example usage
if __name__ == "__main__":
    # Define parameters
    img_size = (96, 96, 96)
    in_channels = 1
    out_channels = 2
    feature_size = 16
    label_nc = 3

    # Create model
    model = SPADEUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        label_nc=label_nc,
    )

    # Test data
    x = torch.randn(1, in_channels, 192, 192, 192)
    segmap = torch.randn(1, label_nc, 192, 192, 192)

    # For inference
    roi_size = (96, 96, 96)  # Smaller than input size

    # Run inference
    with torch.no_grad():
        output = model.inference(x, segmap, roi_size=roi_size)

    print(f"Output shape: {output.shape}")
