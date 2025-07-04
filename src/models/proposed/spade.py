import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)

        nhidden = 64

        # segmap 3x3 conv + relu (label_nc → label_nc)
        self.conv_segmap = nn.Sequential(
            nn.Conv3d(label_nc, label_nc, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # x 1x1 conv + relu (norm_nc → label_nc)
        self.conv_x_feature = nn.Sequential(
            nn.Conv3d(norm_nc, label_nc, kernel_size=1),
            nn.ReLU()
        )
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        # interpolate segmap to the same size as x
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        
        # segmap 3x3 conv + relu
        segmap_feature = self.conv_segmap(segmap)  # [B, label_nc, D, H, W]
        
        # x 1x1 conv + relu  
        x_feature = self.conv_x_feature(x)         # [B, label_nc, D, H, W]
        
        # element-wise sum
        combined_input = segmap_feature + x_feature  # [B, label_nc, D, H, W]
        
        # gamma, beta
        actv = self.mlp_shared(combined_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out    

"""
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
"""