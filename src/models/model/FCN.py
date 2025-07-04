import torch as th
import torch.nn as nn


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class Unet_module(nn.Module):
    def __init__(self, kernel_size, de_kernel_size, channel_list, down_up='down'):
        super(Unet_module, self).__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])
        self.bridge_conv = nn.Conv3d(channel_list[0], channel_list[-1], kernel_size, 1, (kernel_size - 1) // 2)

        if down_up == 'down':
            self.sample = nn.Sequential(
                nn.Conv3d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2, 1),
                nn.BatchNorm3d(channel_list[2]), nn.PReLU())
        else:
            self.sample = nn.Sequential(
                nn.ConvTranspose3d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2),
                nn.BatchNorm3d(channel_list[2]), nn.ReLU())

    def forward(self, x):
        res = self.bridge_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + res
        next_layer = self.sample(x)

        return next_layer, x


class de_conv_module(nn.Module):
    def __init__(self, kernel_size, de_kernel_size, channel_list, down_up='down'):
        super().__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])
        self.bridge_conv = nn.Conv3d(channel_list[0], channel_list[-1], kernel_size, 1, (kernel_size - 1) // 2)

        if down_up == 'down':
            self.sample = nn.Sequential(
                nn.Conv3d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2, 1),
                nn.BatchNorm3d(channel_list[2]), nn.PReLU())
        else:
            self.sample = nn.Sequential(
                nn.ConvTranspose3d(channel_list[2], channel_list[2], de_kernel_size, 2, (de_kernel_size - 1) // 2),
                nn.BatchNorm3d(channel_list[2]), nn.ReLU())

    def forward(self, x, x1):
        x = th.cat([x, x1], dim=1)
        res = self.bridge_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + res
        next_layer = self.sample(x)

        return next_layer


class FCN_Gate(nn.Module):
    '''
    Paper: Coronary Arteries Segmentation Based on 3D FCN With Attention Gate and Level Set Function
    '''
    def __init__(self, channel):
        super().__init__()

        # channel=2
        self.conv1 = nn.Sequential(nn.Conv3d(1, channel, 5, 1, padding=2), nn.BatchNorm3d(channel), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(channel, channel * 2, 2, 2, padding=0), nn.BatchNorm3d(channel * 2),
                                   nn.PReLU())
        self.conv3 = Unet_module(5, 2, [channel * 2, channel * 2, channel * 4], 'down')
        self.conv4 = Unet_module(5, 2, [channel * 4, channel * 4, channel * 8], 'down')
        self.conv5 = Unet_module(5, 2, [channel * 8, channel * 8, channel * 16], 'down')

        self.de_conv1 = Unet_module(5, 2, [channel * 16, channel * 32, channel * 16], down_up='up')
        self.de_conv2 = de_conv_module(5, 2, [channel * 32, channel * 8, channel * 8], down_up='up')
        self.de_conv3 = de_conv_module(5, 2, [channel * 16, channel * 4, channel * 4], down_up='up')
        self.de_conv4 = de_conv_module(5, 2, [channel * 8, channel * 2, channel], down_up='up')

        self.att1 = Attention_block(channel * 16, channel * 16, channel * 8)
        self.last_conv = nn.Conv3d(channel * 2, 1, 1, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = x
        x = self.conv2(x)
        x, x_2 = self.conv3(x)
        x, x_3 = self.conv4(x)
        x, x_4 = self.conv5(x)

        x, _ = self.de_conv1(x)
        x_4 = self.att1(x_4, x)
        x = self.de_conv2(x, x_4)
        x = self.de_conv3(x, x_3)
        x = self.de_conv4(x, x_2)

        x = th.cat([x, x_1], dim=1)
        output = self.last_conv(x)
        return output


class FCN(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # channel=2
        self.conv1 = nn.Sequential(nn.Conv3d(1, channel, 5, 1, padding=2), nn.BatchNorm3d(channel), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(channel, channel * 2, 2, 2, padding=0), nn.BatchNorm3d(channel * 2),
                                   nn.PReLU())
        self.conv3 = Unet_module(5, 2, [channel * 2, channel * 2, channel * 4], 'down')
        self.conv4 = Unet_module(5, 2, [channel * 4, channel * 4, channel * 8], 'down')
        self.conv5 = Unet_module(5, 2, [channel * 8, channel * 8, channel * 16], 'down')

        self.de_conv1 = Unet_module(5, 2, [channel * 16, channel * 32, channel * 16], down_up='up')
        self.de_conv2 = de_conv_module(5, 2, [channel * 32, channel * 8, channel * 8], down_up='up')
        self.de_conv3 = de_conv_module(5, 2, [channel * 16, channel * 4, channel * 4], down_up='up')
        self.de_conv4 = de_conv_module(5, 2, [channel * 8, channel * 2, channel], down_up='up')

        self.last_conv = nn.Conv3d(channel * 2, 1, 1, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x_1 = x
        x = self.conv2(x)
        x, x_2 = self.conv3(x)
        x, x_3 = self.conv4(x)
        x, x_4 = self.conv5(x)

        x, _ = self.de_conv1(x)
        # x_4=self.att1(x_4,x)
        x = self.de_conv2(x, x_4)
        x = self.de_conv3(x, x_3)
        x = self.de_conv4(x, x_2)

        x = th.cat([x, x_1], dim=1)
        output = self.last_conv(x)
        return output


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # net = FCN_Gate_GPU(12)
    net = FCN_Gate(4).cuda()
