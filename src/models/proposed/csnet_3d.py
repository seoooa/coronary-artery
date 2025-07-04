"""
3D Channel and Spatial Attention Network (CSA-Net 3D).
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.proposed.spade import SPADE


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = torch.add(out, residual)
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ModifiedDecoder3d(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc):
        super(ModifiedDecoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
        # SPADE normalization
        self.spade1 = SPADE(out_channels, label_nc)
        self.spade2 = SPADE(out_channels, label_nc)

    def forward(self, x, segmap=None):
        out = self.conv1(x)
        out = self.bn1(out)
        if segmap is not None:
            out = self.spade1(out, segmap)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if segmap is not None:
            out = self.spade2(out, segmap)
        out = self.relu(out)
        
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = torch.add(self.gamma * weights, x)
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = torch.add(self.gamma * weights, x)
        return out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = torch.add(torch.add(sab, cab), x)
        return out


class CSNet3D(nn.Module):
    def __init__(self, classes, channels, label_nc=1):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        :param label_nc: the number of channels in the segmentation map for SPADE normalization.
        """
        super(CSNet3D, self).__init__()
        self.enc_input = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(256)
        self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        
        # Modified decoders with SPADE support
        self.decoder4 = ModifiedDecoder3d(256, 128, label_nc)
        self.decoder3 = ModifiedDecoder3d(128, 64, label_nc)
        self.decoder2 = ModifiedDecoder3d(64, 32, label_nc)
        self.decoder1 = ModifiedDecoder3d(32, 16, label_nc)
        
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        self.final = nn.Conv3d(16, classes, kernel_size=1)
        self.label_nc = label_nc
        initialize_weights(self)

    def forward(self, x, segmap=None):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)
        attention_fuse = torch.add(input_feature, attention)

        # Do decoder operations here with SPADE normalization
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4, segmap)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3, segmap)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2, segmap)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1, None)  # No SPADE for the last decoder

        final = self.final(dec1)
        final = F.softmax(final, dim=1)
        return final

    def inference(self, x_in, segmap=None, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5):
        """
        Perform sliding window inference with proper handling of segmap
        """
        from monai.inferers import sliding_window_inference
        
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
    classes = 2
    channels = 1
    label_nc = 3

    # Create model
    model = CSNet3D(classes=classes, channels=channels, label_nc=label_nc)

    # Test data
    x = torch.randn(1, channels, 96, 96, 96)
    segmap = torch.randn(1, label_nc, 96, 96, 96)

    # Run inference
    with torch.no_grad():
        output = model(x, segmap)
        # Or use sliding window inference
        output_inference = model.inference(x, segmap, roi_size=(96, 96, 96))

    print(f"Input shape: {x.shape}")
    print(f"Segmap shape: {segmap.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference output shape: {output_inference.shape}")