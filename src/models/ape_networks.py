import autorootcwd
import torch
from .proposed.ape_segresnet import APESegResNet
from .proposed.unetr import UNETR
from .proposed.swin_unetr import SwinUNETRv2

class NetworkFactory:
    @staticmethod
    def create_network(arch_name, patch_size=(96, 96, 96), label_nc=8):
        if arch_name == "SegResNet":
            return APESegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
                label_nc=label_nc,
                ape_nc=3,
            )
        
        elif arch_name == "UNETR":
            return UNETR(
                in_channel=1,
                out_channel=2,
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
        
        else:
            raise ValueError(f"Unsupported architecture name: {arch_name}")