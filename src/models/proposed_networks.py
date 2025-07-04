import autorootcwd
import torch
from .proposed.segresnet import SegResNet, PPESegResNet
from .proposed.unetr import UNETR
from .proposed.swin_unetr import SwinUNETRv2
from .proposed.nnformer import nnFormer
from .proposed.csnet_3d import CSNet3D
from .proposed.attentionunet import AttentionUnet
from .proposed.vnet import VNet

class NetworkFactory:
    @staticmethod
    def create_network(arch_name, patch_size=(96, 96, 96), label_nc=8):
        if arch_name == "SegResNet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
                label_nc=label_nc,
            )
        
        elif arch_name == "UNETR":
            return UNETR(
                in_channels=1,
                out_channels=2,
                img_size=patch_size,
                # feature_size=16,
                label_nc=label_nc,
            )
        
        elif arch_name == "SwinUNETR":
            model = SwinUNETRv2(
                img_size=patch_size,
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=True,
                label_nc=label_nc,
            )
            weight = torch.load("weight/model_swinvit.pt", weights_only=True)
            
            # Extract only the SwinTransformer (swinViT) weights
            swin_vit_weights = {
                k: v for k, v in weight.items() if k.startswith("swinViT")
            }

            # Load the SwinTransformer weights into the model
            model.swinViT.load_state_dict(swin_vit_weights, strict=False)
            print("Using pretrained self-supervised Swin UNETR SwinTransformer weights!")
            return model
        
        elif arch_name == "nnFormer":
            return nnFormer(
                crop_size=patch_size,
                embedding_dim=192,
                input_channels=1,
                num_classes=2,
                depths=[2, 2, 2, 2],
                num_heads=[6, 12, 24, 48],
                patch_size=[2, 4, 4],
                window_size=[4, 4, 8, 4],
                deep_supervision=False,
                label_nc=label_nc,
            )
        
        elif arch_name == "PPESegResNet":
            return PPESegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
            )
        
        elif arch_name == "CSNet3D":
            return CSNet3D(
                classes=2,
                channels=1,
                label_nc=label_nc,
            )
        
        elif arch_name == "AttentionUnet":
            return AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                dropout=0.1,
                label_nc=label_nc,
            )
        
        elif arch_name == "VNet":
            return VNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                act="relu",
                dropout_prob_down=0.5,
                dropout_prob_up=(0.5, 0.5),
                label_nc=label_nc,
            )
        
        else:
            raise ValueError(f"Unsupported architecture name: {arch_name}")