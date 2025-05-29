import torch
from monai.networks.nets import UNet, AttentionUnet, SegResNet, UNETR, SwinUNETR, VNet
from monai.networks.layers import Norm

class NetworkFactory:
    @staticmethod
    def create_network(arch_name, patch_size=(96, 96, 96)):
        if arch_name == "UNet":
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            )
        
        elif arch_name == "AttentionUnet":
            return AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                dropout=0.1,
            )
        
        elif arch_name == "SegResNet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
            )
        
        elif arch_name == "UNETR":
            return UNETR(
                in_channels=1,
                out_channels=2,
                img_size=patch_size,
                feature_size=16
            )
        
        elif arch_name == "SwinUNETR":
            model = SwinUNETR(
                img_size=patch_size,
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=True,
            )
            weight = torch.load("weight/model_swinvit.pt", weights_only=True)
            model.load_from(weight)
            print("Using pretrained self-supervised Swin UNETR backbone weights!")
            return model
        
        elif arch_name == "VNet":
            return VNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                act="relu",
                dropout_prob=0.5,
            )
        
        else:
            raise ValueError(f"Unsupported architecture name: {arch_name}")