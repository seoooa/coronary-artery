from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch as th
import torch


class Unet_module(nn.Module):
    def __init__(self, kernel_size, channel_list, down_up='down'):
        super(Unet_module, self).__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])

        if down_up == 'down':
            self.sample = nn.MaxPool3d(2, 2)
        else:
            self.sample = nn.Sequential(nn.ConvTranspose3d(channel_list[2], channel_list[2],
                                                           kernel_size, 2, (kernel_size - 1) // 2, 1), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        next_layer = self.sample(x)

        return next_layer, x


class Unet(nn.Module):
    '''
    The network structure of "Coronary Artery Segmentation in Cardiac CT Angiography Using 3D Multi-Channel U-net"
    '''
    def __init__(self,in_channel,kernel_size):
        super(Unet, self).__init__()

        self.u1 = Unet_module(kernel_size, (in_channel, 32, 64))
        self.u2 = Unet_module(kernel_size, (64, 64, 128))
        self.u3 = Unet_module(kernel_size, (128, 128, 256), down_up='up')
        self.u4 = Unet_module(kernel_size, (384, 128, 128), down_up='up')
        self.u5 = Unet_module(kernel_size, (192, 64, 64), down_up='up')
        self.last_conv = nn.Conv3d(64, 1, 1, 1,bias=False)
        # self.activate_fun=nn.Sigmoid()

    def forward(self, x):
        x, x_c1 = self.u1(x)
        x, x_c2 = self.u2(x)
        x, x1 = self.u3(x)
        x = th.cat([x, x_c2], dim=1)
        x, _ = self.u4(x)
        x = th.cat([x, x_c1], dim=1)
        _, x = self.u5(x)
        x = self.last_conv(x)
        # x=self.activate_fun(x)
        return x


class Unet_Patch(nn.Module):
    def __init__(self, kernel_size,in_channel=2):
        super(Unet_Patch, self).__init__()

        self.u1 = Unet_module(kernel_size, (in_channel, 32, 64))
        self.u2 = Unet_module(kernel_size, (64, 64, 128))
        self.u3 = Unet_module(kernel_size, (128, 128, 256), down_up='up')
        self.u4 = Unet_module(kernel_size, (384, 128, 128), down_up='up')
        self.u5 = Unet_module(kernel_size, (192, 64, 64), down_up='up')
        self.u6=nn.Sequential(nn.Conv3d(64,32,3,1,1,bias=False),nn.ReLU(inplace=True),nn.BatchNorm3d(32))
        self.last_conv = nn.Conv3d(32, 1, 1, 1,bias=False)
        # self.activate_fun=nn.Sigmoid()
        self.dropout1=nn.Dropout3d(0.5)
        self.dropout2=nn.Dropout3d(0.5)

    def forward(self, x):
        x, x_c1 = self.u1(x)
        x, x_c2 = self.u2(x)
        x, x1 = self.u3(x)
        x = th.cat([x, x_c2], dim=1)
        x, _ = self.u4(x)
        x = th.cat([x, x_c1], dim=1)
        _, x = self.u5(x)
        x=self.u6(x)
        x=self.dropout1(x)
        x = self.last_conv(x)
        return x


class conv_block_nested3d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested3d, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet3d(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1,channel=16):
        super(NestedUNet3d, self).__init__()

        n1 = channel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.Up = nn.Upsample(scale_factor=2)

        self.conv0_0 = conv_block_nested3d(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested3d(filters[0], filters[1], filters[1])
        self.conv2_0 =conv_block_nested3d(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested3d(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested3d(filters[3], filters[4], filters[4])

        self.conv0_1 =conv_block_nested3d(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 =conv_block_nested3d(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested3d(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested3d(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested3d(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested3d(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested3d(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested3d(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested3d(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested3d(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


