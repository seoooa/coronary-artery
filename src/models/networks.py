import torch
from monai.networks.nets import UNet, AttentionUnet, SegResNet, UNETR, SwinUNETR, VNet
from monai.networks.layers import Norm
from src.models.model.nnformer import nnFormer
from src.models.model.csnet_3d import CSNet3D
from src.models.model.DSCNet import DSCNet

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
                dropout_prob_down=0.5,
                dropout_prob_up=(0.5, 0.5),
            )
        
        elif arch_name == "nnFormer":
            return nnFormer(
                crop_size=[96, 96, 96],
                embedding_dim=96,
                input_channels=1,
                num_classes=2,
                conv_op=torch.nn.Conv3d,
                depths=[2, 2, 2, 2],
                num_heads=[6, 12, 24, 48],
                patch_size=[2, 4, 4],
                window_size=[6, 6, 6, 6],
                deep_supervision=True
            )
        
        elif arch_name == "CSNet3D":
            return CSNet3D(
                classes=2,
                channels=1
            )
        
        elif arch_name == "DSCNet":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return DSCNet(
                n_channels=1,
                n_classes=2,
                kernel_size=3,
                extend_scope=1.0,
                if_offset=True,
                device=device,
                number=4,
                dim=3
            )
        
        else:
            raise ValueError(f"Unsupported architecture name: {arch_name}")