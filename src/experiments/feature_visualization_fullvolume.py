"""
SPADE Feature Visualization - Full Volume Version
전체 볼륨에 대해 sliding window로 feature를 추출하고 합치는 버전
"""

import autorootcwd
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import click
import os
from pathlib import Path
from tqdm import tqdm
import json

# Import models
from src.models.proposed.segresnet import SegResNet as ProposedSegResNet
from src.models.model.segresnet import SegResNet as BaselineSegResNet
from monai.inferers import sliding_window_inference

# Import transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    AsDiscreted,
    Lambda,
)
from src.data.proposed_dataloader import ConvertDistanceMap


class FeatureExtractor:
    """Hook을 사용하여 중간 feature 추출"""
    
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def get_hook(self, name):
        """특정 layer의 output을 저장하는 hook 함수"""
        def hook(module, input, output):
            # segmap과 함께 들어오는 경우 처리
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self, model, layer_names):
        """모델의 특정 layer에 hook 등록"""
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """등록된 hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}


def get_transforms():
    """Baseline과 Proposed용 transform 생성"""
    # Baseline transform (seg 없음)
    baseline_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-150,
            a_max=550,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])
    
    # Proposed transform (seg 포함, distance map 변환)
    proposed_transform = Compose([
        LoadImaged(keys=["image", "label", "seg"]),
        EnsureChannelFirstd(keys=["image", "label", "seg"]),
        Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-150,
            a_max=550,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        AsDiscreted(
            keys=["seg"],
            to_onehot=8,  # 8채널 one-hot 인코딩
        ),
        CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        Lambda(ConvertDistanceMap),  # 8채널 distance map 변환
    ])
    
    return baseline_transform, proposed_transform


def load_models(baseline_ckpt, proposed_ckpt, device):
    """Baseline과 Proposed 모델 로드"""
    print("Loading models...")
    
    # Baseline model (SPADE 없음)
    baseline_model = BaselineSegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )
    
    # Proposed model (SPADE 있음)
    proposed_model = ProposedSegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        label_nc=8,  # 8채널 distance map
    )
    
    # Load checkpoints
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        baseline_state = torch.load(baseline_ckpt, map_location=device)
        if 'state_dict' in baseline_state:
            # Lightning checkpoint
            state_dict = {k.replace('_model.', ''): v for k, v in baseline_state['state_dict'].items() if k.startswith('_model.')}
            baseline_model.load_state_dict(state_dict)
        else:
            baseline_model.load_state_dict(baseline_state)
        print(f"✓ Loaded baseline model from {baseline_ckpt}")
    
    if proposed_ckpt and os.path.exists(proposed_ckpt):
        proposed_state = torch.load(proposed_ckpt, map_location=device)
        if 'state_dict' in proposed_state:
            # Lightning checkpoint
            state_dict = {k.replace('_model.', ''): v for k, v in proposed_state['state_dict'].items() if k.startswith('_model.')}
            proposed_model.load_state_dict(state_dict)
        else:
            proposed_model.load_state_dict(proposed_state)
        print(f"✓ Loaded proposed model from {proposed_ckpt}")
    
    baseline_model.to(device).eval()
    proposed_model.to(device).eval()
    
    return baseline_model, proposed_model


def compute_importance_map(patch_size, mode="gaussian", sigma_scale=0.125, device="cpu"):
    """
    3D Gaussian importance map 생성 (MONAI와 동일한 방식)
    
    Args:
        patch_size: tuple (d, h, w)
        mode: "gaussian" or "constant"
        sigma_scale: Gaussian sigma의 스케일 (default: 0.125)
        device: torch device
        
    Returns:
        importance_map: [1, 1, d, h, w] tensor
    """
    if mode == "constant":
        return torch.ones((1, 1, *patch_size), device=device)
    
    # Gaussian importance map
    center = np.array(patch_size) / 2.0
    sigma = sigma_scale * np.array(patch_size)
    
    # Create coordinate grid
    coords = np.stack(np.meshgrid(
        np.arange(patch_size[0]),
        np.arange(patch_size[1]),
        np.arange(patch_size[2]),
        indexing='ij'
    ), axis=-1)
    
    # Calculate Gaussian weights
    distances = np.linalg.norm((coords - center) / sigma, axis=-1)
    importance_map = np.exp(-0.5 * distances ** 2)
    
    # Normalize to [0, 1]
    importance_map = importance_map / importance_map.max()
    
    return torch.from_numpy(importance_map).float().unsqueeze(0).unsqueeze(0).to(device)


def sliding_window_feature_inference(model, image, segmap, roi_size, overlap, device, layer_names, use_segmap=False, mode="gaussian"):
    """
    Sliding window로 전체 볼륨을 처리하며 feature map을 추출하고 Gaussian weighted averaging으로 합침
    
    Args:
        model: 모델
        image: [B, C, D, H, W] input image
        segmap: [B, C, D, H, W] segmentation map (for proposed model)
        roi_size: tuple (d, h, w) - patch size
        overlap: float - overlap ratio (0~1)
        device: torch device
        layer_names: list of layer names to extract features from
        use_segmap: bool - whether to use segmap
        mode: "gaussian" or "constant" - weighting mode for overlapping regions
        
    Returns:
        features: dict of {layer_name: [B, C, D, H, W]} - aggregated feature maps
    """
    extractor = FeatureExtractor()
    extractor.register_hooks(model, layer_names)
    
    B, C, D, H, W = image.shape
    roi_d, roi_h, roi_w = roi_size
    
    # Calculate stride based on overlap
    stride_d = int(roi_d * (1 - overlap))
    stride_h = int(roi_h * (1 - overlap))
    stride_w = int(roi_w * (1 - overlap))
    
    print(f"  Using {mode} weighted averaging")
    print(f"  Overlap: {overlap}, Stride: ({stride_d}, {stride_h}, {stride_w})")
    
    # Generate patch coordinates
    starts_d = list(range(0, D - roi_d + 1, stride_d))
    starts_h = list(range(0, H - roi_h + 1, stride_h))
    starts_w = list(range(0, W - roi_w + 1, stride_w))
    
    # Ensure we cover the entire volume
    if starts_d[-1] + roi_d < D:
        starts_d.append(D - roi_d)
    if starts_h[-1] + roi_h < H:
        starts_h.append(H - roi_h)
    if starts_w[-1] + roi_w < W:
        starts_w.append(W - roi_w)
    
    total_patches = len(starts_d) * len(starts_h) * len(starts_w)
    print(f"  Total patches: {total_patches} ({len(starts_d)}x{len(starts_h)}x{len(starts_w)})")
    
    # Initialize feature accumulation dictionaries
    feature_sums = {}
    feature_weights = {}
    
    # Compute importance map for Gaussian weighting (on CPU initially)
    importance_map_cpu = compute_importance_map(roi_size, mode=mode, device="cpu")
    
    with torch.no_grad():
        patch_idx = 0
        for d_start in tqdm(starts_d, desc="  Depth", leave=False):
            for h_start in starts_h:
                for w_start in starts_w:
                    # Extract patch
                    d_end = d_start + roi_d
                    h_end = h_start + roi_h
                    w_end = w_start + roi_w
                    
                    image_patch = image[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)
                    
                    # Forward pass
                    if use_segmap:
                        segmap_patch = segmap[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)
                        _ = model(image_patch, segmap_patch)
                    else:
                        _ = model(image_patch)
                    
                    # Get extracted features for this patch
                    patch_features = extractor.features.copy()
                    
                    # Initialize feature maps on first patch
                    if patch_idx == 0:
                        for layer_name, feat in patch_features.items():
                            _, C_feat, D_feat, H_feat, W_feat = feat.shape
                            
                            # Calculate output spatial size based on feature resolution
                            scale_d = D_feat / roi_d
                            scale_h = H_feat / roi_h
                            scale_w = W_feat / roi_w
                            
                            out_d = int(D * scale_d)
                            out_h = int(H * scale_h)
                            out_w = int(W * scale_w)
                            
                            feature_sums[layer_name] = torch.zeros(B, C_feat, out_d, out_h, out_w)
                            feature_weights[layer_name] = torch.zeros(B, 1, out_d, out_h, out_w)
                    
                    # Accumulate features with Gaussian weighting
                    for layer_name, feat in patch_features.items():
                        _, C_feat, D_feat, H_feat, W_feat = feat.shape
                        
                        # Calculate corresponding coordinates in feature space
                        scale_d = D_feat / roi_d
                        scale_h = H_feat / roi_h
                        scale_w = W_feat / roi_w
                        
                        feat_d_start = int(d_start * scale_d)
                        feat_h_start = int(h_start * scale_h)
                        feat_w_start = int(w_start * scale_w)
                        
                        feat_d_end = feat_d_start + D_feat
                        feat_h_end = feat_h_start + H_feat
                        feat_w_end = feat_w_start + W_feat
                        
                        # Resize importance map to feature resolution
                        if (D_feat, H_feat, W_feat) != roi_size:
                            importance_map_resized = torch.nn.functional.interpolate(
                                importance_map_cpu,
                                size=(D_feat, H_feat, W_feat),
                                mode='trilinear',
                                align_corners=True
                            )
                        else:
                            importance_map_resized = importance_map_cpu
                        
                        # Apply Gaussian weight
                        weighted_feat = feat * importance_map_resized
                        
                        # Accumulate
                        feature_sums[layer_name][:, :, feat_d_start:feat_d_end, feat_h_start:feat_h_end, feat_w_start:feat_w_end] += weighted_feat
                        feature_weights[layer_name][:, :, feat_d_start:feat_d_end, feat_h_start:feat_h_end, feat_w_start:feat_w_end] += importance_map_resized
                    
                    patch_idx += 1
    
    # Weighted average of overlapping regions
    features = {}
    for layer_name in feature_sums.keys():
        features[layer_name] = feature_sums[layer_name] / feature_weights[layer_name].clamp(min=1e-8)
    
    extractor.remove_hooks()
    
    return features


def extract_features_from_patient_fullvolume(model, image, segmap, device, patch_size=(96, 96, 96), overlap=0.25, use_segmap=False, mode="gaussian"):
    """
    환자 데이터에서 전체 볼륨에 대한 feature 추출 (sliding window with Gaussian weighted averaging)
    
    Args:
        model: 모델
        image: input image tensor
        segmap: segmentation map tensor
        device: torch device
        patch_size: patch size for sliding window
        overlap: overlap ratio (default: 0.25, same as training)
        use_segmap: whether to use segmentation map
        mode: "gaussian" or "constant" - weighting mode (default: "gaussian", same as training)
        
    Returns:
        features: dict of extracted features for full volume
    """
    layer_names = ['up_layers.0', 'up_layers.1', 'up_layers.2']
    
    print(f"  Input volume shape: {image.shape}")
    print(f"  Patch size: {patch_size}, Overlap: {overlap}")
    
    features = sliding_window_feature_inference(
        model, image, segmap, 
        roi_size=patch_size, 
        overlap=overlap, 
        device=device,
        layer_names=layer_names,
        use_segmap=use_segmap,
        mode=mode
    )
    
    print(f"  Output feature shapes:")
    for layer_name, feat in features.items():
        print(f"    {layer_name}: {feat.shape}")
    
    return features


def visualize_feature_maps(baseline_features, proposed_features, output_dir, patient_id, 
                           ct_image=None, segmap=None, label=None, seg_original=None):
    """Feature map 시각화 (각 decoder layer별) with CT, segmentation, and label overlay"""
    print(f"\nVisualizing feature maps for patient {patient_id}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Anatomical region colormap and labels
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    region_colors = ['black', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    region_cmap = ListedColormap(region_colors)
    
    region_labels = {
        0: 'Background',
        1: 'Coronary Arteries',
        2: 'Aorta',
        3: 'Myocardium',
        4: 'Left Ventricle',
        5: 'Right Ventricle',
        6: 'Left Atrium',
        7: 'Right Atrium'
    }
    
    for layer_name in baseline_features.keys():
        if layer_name not in proposed_features:
            continue
        
        baseline_feat = baseline_features[layer_name][0]  # [C, D, H, W]
        proposed_feat = proposed_features[layer_name][0]  # [C, D, H, W]
        
        # 중간 slice 선택
        mid_slice = baseline_feat.shape[3] // 2
        
        # 각 feature map의 평균 (채널 평균)
        baseline_slice = baseline_feat[:, :, :, mid_slice].mean(dim=0).numpy()
        proposed_slice = proposed_feat[:, :, :, mid_slice].mean(dim=0).numpy()
        
        # 차이 계산
        diff = proposed_slice - baseline_slice
        
        # CT image와 segmentation을 feature map 크기에 맞게 resize
        if ct_image is not None and seg_original is not None:
            from scipy.ndimage import zoom
            
            # CT slice
            ct_mid_slice = ct_image.shape[4] // 2
            ct_slice = ct_image[0, 0, :, :, ct_mid_slice].cpu().numpy()
            
            # Segmentation slice (use original labels)
            seg_mid_slice = seg_original.shape[4] // 2
            seg_slice = seg_original[0, 0, :, :, seg_mid_slice].cpu().numpy()
            
            # Label slice (coronary artery ground truth)
            if label is not None:
                label_mid_slice = label.shape[4] // 2
                label_slice = label[0, 0, :, :, label_mid_slice].cpu().numpy()
            else:
                label_slice = None
            
            # Resize CT and labels to feature map size for consistent visualization
            target_size = baseline_slice.shape
            zoom_factors = (target_size[0] / ct_slice.shape[0], target_size[1] / ct_slice.shape[1])
            ct_resized = zoom(ct_slice, zoom_factors, order=1)  # bilinear interpolation
            
            # Pre-compute resized versions for Row 3 overlays
            if label_slice is not None:
                label_resized_for_overlay = zoom(label_slice, zoom_factors, order=0)  # nearest neighbor
            
            # 시각화 (3x3 layout with anatomical context)
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))

            print(f"CT image shape: {ct_image.shape}")
            print(f"CT slice shape: {ct_slice.shape}")
            print(f"baseline_feat shape: {baseline_feat.shape}")
            print(f"mid_slice: {mid_slice}, ct_mid_slice: {ct_mid_slice}, seg_mid_slice: {seg_mid_slice}")
            
            # ========== Row 1: Anatomical Context ==========
            # CT image (resize to feature map size for consistent display)
            ct_slice_resized = zoom(ct_slice, zoom_factors, order=1)
            axes[0, 0].imshow(ct_slice_resized, cmap='gray', interpolation='bilinear')
            axes[0, 0].set_title(f'CT Image\nMid-slice {ct_mid_slice} (resized to feature size)', fontsize=11, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Heart segmentation overlay on CT (resize to feature map size)
            seg_slice_resized = zoom(seg_slice, zoom_factors, order=0)  # nearest neighbor for labels
            axes[0, 1].imshow(ct_slice_resized, cmap='gray', interpolation='bilinear')
            seg_overlay_resized = np.ma.masked_where(seg_slice_resized == 0, seg_slice_resized)
            axes[0, 1].imshow(seg_overlay_resized, cmap=region_cmap, alpha=0.5, vmin=0, vmax=7, interpolation='nearest')
            axes[0, 1].set_title(f'Heart Segmentation\n(8 Anatomical Regions)', fontsize=11, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Add legend for anatomical regions (only for present labels)
            unique_labels = np.unique(seg_slice_resized[seg_slice_resized > 0])  # Exclude background
            legend_elements = [Patch(facecolor=region_colors[int(label)], label=f'{int(label)}: {region_labels[int(label)]}') 
                             for label in unique_labels]
            axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.8)
            
            # Coronary artery label (ground truth, resize to feature map size)
            if label_slice is not None:
                label_slice_resized = zoom(label_slice, zoom_factors, order=0)  # nearest neighbor
                axes[0, 2].imshow(ct_slice_resized, cmap='gray', interpolation='bilinear')
                label_overlay_resized = np.ma.masked_where(label_slice_resized == 0, label_slice_resized)
                axes[0, 2].imshow(label_overlay_resized, cmap='Reds', alpha=0.7, interpolation='nearest')
                axes[0, 2].set_title(f'Coronary Artery Label\n(Ground Truth, resized)', fontsize=11, fontweight='bold')
            else:
                axes[0, 2].text(0.5, 0.5, 'Label Not Available', ha='center', va='center', 
                               fontsize=12, transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
            
            # ========== Row 2: Feature Maps ==========
            # Baseline
            im1 = axes[1, 0].imshow(baseline_slice, cmap='viridis')
            axes[1, 0].set_title(f'Baseline Features\n{layer_name}', fontsize=11, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
            
            # Proposed
            im2 = axes[1, 1].imshow(proposed_slice, cmap='viridis')
            axes[1, 1].set_title(f'Proposed (SPADE) Features\n{layer_name}', fontsize=11, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
            
            # Difference (Proposed - Baseline)
            im3 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[1, 2].set_title(f'Difference\n(Proposed - Baseline)', fontsize=11, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
            
            # ========== Row 3: Overlays and Analysis ==========
            # Absolute difference
            im4 = axes[2, 0].imshow(np.abs(diff), cmap='hot')
            axes[2, 0].set_title(f'Absolute Difference', fontsize=11, fontweight='bold')
            axes[2, 0].axis('off')
            plt.colorbar(im4, ax=axes[2, 0], fraction=0.046)
            
            # Overlay: Absolute difference on CT
            axes[2, 1].imshow(ct_resized, cmap='gray', alpha=0.7)
            diff_overlay = np.ma.masked_where(np.abs(diff) < np.percentile(np.abs(diff), 50), np.abs(diff))
            im5 = axes[2, 1].imshow(diff_overlay, cmap='hot', alpha=0.8)
            axes[2, 1].set_title(f'Feature Change Overlay\n(on CT)', fontsize=11, fontweight='bold')
            axes[2, 1].axis('off')
            plt.colorbar(im5, ax=axes[2, 1], fraction=0.046)
            
            # Overlay: Feature change with segmentation context
            if label_slice is not None:
                axes[2, 2].imshow(ct_resized, cmap='gray', alpha=0.6)
                # Show coronary artery region
                coronary_mask = np.ma.masked_where(label_resized_for_overlay == 0, label_resized_for_overlay)
                axes[2, 2].imshow(coronary_mask, cmap='Greens', alpha=0.4, interpolation='nearest')
                # Overlay feature change
                axes[2, 2].imshow(diff_overlay, cmap='hot', alpha=0.6)
                axes[2, 2].set_title(f'Feature Change\n(with Coronary Context)', fontsize=11, fontweight='bold')
            else:
                axes[2, 2].imshow(ct_resized, cmap='gray', alpha=0.7)
                seg_overlay_resized = np.ma.masked_where(seg_slice_resized == 0, seg_slice_resized)
                axes[2, 2].imshow(seg_overlay_resized, cmap=region_cmap, alpha=0.3, vmin=0, vmax=7, interpolation='nearest')
                axes[2, 2].imshow(diff_overlay, cmap='hot', alpha=0.6)
                axes[2, 2].set_title(f'Feature Change\n(with Anatomical Context)', fontsize=11, fontweight='bold')
                # Add small legend
                unique_labels_subset = np.unique(seg_slice_resized[seg_slice_resized > 0])[:4]  # Show first 4 labels only
                legend_elements_small = [Patch(facecolor=region_colors[int(label)], label=region_labels[int(label)][:10]) 
                                        for label in unique_labels_subset]
                axes[2, 2].legend(handles=legend_elements_small, loc='upper right', fontsize=6, framealpha=0.7)
            axes[2, 2].axis('off')
            
        else:
            # Original 4-panel layout (fallback)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            im0 = axes[0].imshow(baseline_slice, cmap='viridis')
            axes[0].set_title(f'Baseline\n{layer_name}')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(proposed_slice, cmap='viridis')
            axes[1].set_title(f'Proposed (SPADE)\n{layer_name}')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[2].set_title(f'Difference\n{layer_name}')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])
            
            im3 = axes[3].imshow(np.abs(diff), cmap='hot')
            axes[3].set_title(f'Absolute Difference\n{layer_name}')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3])
        
        plt.suptitle(f'Feature Maps Comparison - Patient {patient_id} (Full Volume)', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top for suptitle
        
        # Save
        safe_layer_name = layer_name.replace('.', '_')
        save_path = os.path.join(output_dir, f'patient_{patient_id}_layer_{safe_layer_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")


def sample_features_by_anatomy(features, segmap_original, num_samples=1000, exclude_background=True):
    """
    해부학적 위치별로 feature 샘플링
    
    Args:
        features: dict of {layer_name: tensor [B, C, D, H, W]}
        segmap_original: tensor [B, 1, D, H, W] - 해부학적 레이블
        num_samples: 각 해부학적 위치당 샘플 수
        exclude_background: background (label 0) 제외 여부
    
    Returns:
        sampled_features: [N, C] - 샘플링된 feature 벡터
        labels: [N] - 해부학적 위치 레이블
    """
    # 마지막 decoder layer의 feature 사용 (가장 고해상도)
    layer_names = sorted(features.keys())
    last_layer = layer_names[-1]
    feat = features[last_layer][0]  # [C, D, H, W]
    
    # Segmap을 feature와 같은 크기로 resize
    segmap_resized = torch.nn.functional.interpolate(
        segmap_original,
        size=feat.shape[1:],
        mode='nearest'
    )[0, 0].numpy()  # [D, H, W]
    
    # Feature를 [C, N] 형태로 변환
    C, D, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).numpy()  # [C, N]
    segmap_flat = segmap_resized.reshape(-1)  # [N]
    
    # 각 해부학적 위치별로 샘플링
    sampled_features = []
    labels = []
    
    unique_labels = np.unique(segmap_flat)
    if exclude_background:
        unique_labels = unique_labels[unique_labels > 0]  # background 제외
    
    for label_id in unique_labels:
        indices = np.where(segmap_flat == label_id)[0]
        
        if len(indices) > num_samples:
            # 랜덤 샘플링
            sampled_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            # 모든 voxel 사용
            sampled_indices = indices
        
        sampled_features.append(feat_flat[:, sampled_indices].T)  # [N_sample, C]
        labels.extend([label_id] * len(sampled_indices))
    
    sampled_features = np.concatenate(sampled_features, axis=0)  # [N_total, C]
    labels = np.array(labels)  # [N_total]
    
    return sampled_features, labels


def sample_features_by_anatomy_binary(features, segmap_original, target_label, num_samples=1000):
    """
    특정 레이블 vs 나머지로 feature 샘플링 (Binary classification)
    
    Args:
        features: dict of {layer_name: tensor [B, C, D, H, W]}
        segmap_original: tensor [B, 1, D, H, W] - 해부학적 레이블
        target_label: 타겟 레이블 (1-7)
        num_samples: 각 클래스당 샘플 수
    
    Returns:
        sampled_features: [N, C] - 샘플링된 feature 벡터
        labels: [N] - Binary 레이블 (0: others, 1: target)
    """
    # 마지막 decoder layer의 feature 사용
    layer_names = sorted(features.keys())
    last_layer = layer_names[-1]
    feat = features[last_layer][0]  # [C, D, H, W]
    
    # Segmap을 feature와 같은 크기로 resize
    segmap_resized = torch.nn.functional.interpolate(
        segmap_original,
        size=feat.shape[1:],
        mode='nearest'
    )[0, 0].numpy()  # [D, H, W]
    
    # Feature를 [C, N] 형태로 변환
    C, D, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).numpy()  # [C, N]
    segmap_flat = segmap_resized.reshape(-1)  # [N]
    
    # Binary mask 생성 (target vs others, background 제외)
    target_mask = segmap_flat == target_label
    others_mask = (segmap_flat != target_label) & (segmap_flat > 0)  # background 제외
    
    sampled_features = []
    labels = []
    
    # Sample from target label
    target_indices = np.where(target_mask)[0]
    if len(target_indices) > num_samples:
        sampled_target_indices = np.random.choice(target_indices, num_samples, replace=False)
    else:
        sampled_target_indices = target_indices
    
    sampled_features.append(feat_flat[:, sampled_target_indices].T)
    labels.extend([1] * len(sampled_target_indices))  # Target = 1
    
    # Sample from others
    others_indices = np.where(others_mask)[0]
    if len(others_indices) > num_samples:
        sampled_others_indices = np.random.choice(others_indices, num_samples, replace=False)
    else:
        sampled_others_indices = others_indices
    
    sampled_features.append(feat_flat[:, sampled_others_indices].T)
    labels.extend([0] * len(sampled_others_indices))  # Others = 0
    
    sampled_features = np.concatenate(sampled_features, axis=0)  # [N_total, C]
    labels = np.array(labels)  # [N_total]
    
    return sampled_features, labels


def visualize_tsne(baseline_features, proposed_features, baseline_labels, proposed_labels, 
                   output_dir, patient_id, perplexity=30, suffix="heart_all"):
    """t-SNE 시각화 (2D + 3D)
    
    Args:
        suffix: 
            - "heart_all": All 7 heart anatomical regions
            - "heart_{label_name}": Binary (target label vs others)
    """
    print(f"\nComputing t-SNE for patient {patient_id} ({suffix})...")
    
    # Anatomical region names and title
    if suffix == "heart_all":
        label_names = {
            1: 'Coronary Arteries',
            2: 'Aorta',
            3: 'Myocardium',
            4: 'Left Ventricle',
            5: 'Right Ventricle',
            6: 'Left Atrium',
            7: 'Right Atrium'
        }
        title_suffix = "All Heart Anatomical Regions"
    elif suffix.startswith("heart_"):
        # Binary: target label vs others
        label_name_parts = suffix.replace("heart_", "").replace("_", " ").title()
        label_names = {
            0: 'Others',
            1: label_name_parts
        }
        title_suffix = f"{label_name_parts} vs Others"
    else:
        label_names = {0: 'Unknown'}
        title_suffix = suffix
    
    # PCA로 차원 먼저 축소 (t-SNE 속도 향상)
    print("  Applying PCA preprocessing...")
    pca = PCA(n_components=min(50, baseline_features.shape[1]))
    baseline_pca = pca.fit_transform(baseline_features)
    proposed_pca = pca.transform(proposed_features)
    
    # ========== 2D t-SNE ==========
    print("  Computing 2D t-SNE for baseline...")
    tsne_baseline_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    baseline_2d = tsne_baseline_2d.fit_transform(baseline_pca)
    
    print("  Computing 2D t-SNE for proposed...")
    tsne_proposed_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    proposed_2d = tsne_proposed_2d.fit_transform(proposed_pca)
    
    # ========== 3D t-SNE ==========
    print("  Computing 3D t-SNE for baseline...")
    tsne_baseline_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_jobs=-1)
    baseline_3d = tsne_baseline_3d.fit_transform(baseline_pca)
    
    print("  Computing 3D t-SNE for proposed...")
    tsne_proposed_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_jobs=-1)
    proposed_3d = tsne_proposed_3d.fit_transform(proposed_pca)
    
    # Discrete colormap
    unique_labels = np.unique(baseline_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # ========== 2D Visualization ==========
    print("  Creating 2D visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Baseline 2D
    for label in unique_labels:
        mask = baseline_labels == label
        # Color index: use label directly for binary, label-1 for multi-class
        if suffix.startswith("heart_") and suffix != "heart_all":
            label_idx = int(label)  # Binary: 0 or 1
        else:
            label_idx = int(label) - 1  # Multi-class: 1-7 → 0-6
        
        axes[0].scatter(
            baseline_2d[mask, 0], baseline_2d[mask, 1],
            c=[colors[label_idx]], s=10, alpha=0.6,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    axes[0].set_title(f'Baseline Model\nPatient {patient_id}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Proposed 2D
    legend_handles = []
    legend_labels = []
    for label in unique_labels:
        mask = proposed_labels == label
        # Color index: use label directly for binary, label-1 for multi-class
        if suffix.startswith("heart_") and suffix != "heart_all":
            label_idx = int(label)  # Binary: 0 or 1
        else:
            label_idx = int(label) - 1  # Multi-class: 1-7 → 0-6
        
        h = axes[1].scatter(
            proposed_2d[mask, 0], proposed_2d[mask, 1],
            c=[colors[label_idx]], s=10, alpha=0.6,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
        legend_handles.append(h)
        legend_labels.append(f'{int(label)}: {label_names.get(int(label), "Unknown")}')
    
    axes[1].set_title(f'Proposed Model (SPADE)\nPatient {patient_id}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 공통 Legend를 그래프 아래에 배치
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=4, 
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='Anatomical Regions',
              title_fontsize=11)
    
    plt.suptitle(f'Feature Clustering {title_suffix} (2D t-SNE) - Full Volume', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save 2D
    save_path_2d = os.path.join(output_dir, f'patient_{patient_id}_tsne_2d_{suffix}.png')
    plt.savefig(save_path_2d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 2D: {save_path_2d}")
    
    # ========== 3D Visualization ==========
    print("  Creating 3D visualization...")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 9))
    
    # Baseline 3D
    ax1 = fig.add_subplot(121, projection='3d')
    for label in unique_labels:
        mask = baseline_labels == label
        # Color index: use label directly for binary, label-1 for multi-class
        if suffix.startswith("heart_") and suffix != "heart_all":
            label_idx = int(label)  # Binary: 0 or 1
        else:
            label_idx = int(label) - 1  # Multi-class: 1-7 → 0-6
        
        ax1.scatter(
            baseline_3d[mask, 0], baseline_3d[mask, 1], baseline_3d[mask, 2],
            c=[colors[label_idx]], s=10, alpha=0.6,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax1.set_title(f'Baseline Model\nPatient {patient_id}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('t-SNE Component 1', fontsize=11)
    ax1.set_ylabel('t-SNE Component 2', fontsize=11)
    ax1.set_zlabel('t-SNE Component 3', fontsize=11)
    ax1.view_init(elev=20, azim=45)  # Set viewing angle
    
    # Proposed 3D
    ax2 = fig.add_subplot(122, projection='3d')
    for label in unique_labels:
        mask = proposed_labels == label
        # Color index: use label directly for binary, label-1 for multi-class
        if suffix.startswith("heart_") and suffix != "heart_all":
            label_idx = int(label)  # Binary: 0 or 1
        else:
            label_idx = int(label) - 1  # Multi-class: 1-7 → 0-6
        
        ax2.scatter(
            proposed_3d[mask, 0], proposed_3d[mask, 1], proposed_3d[mask, 2],
            c=[colors[label_idx]], s=10, alpha=0.6,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax2.set_title(f'Proposed Model (SPADE)\nPatient {patient_id}', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('t-SNE Component 1', fontsize=11)
    ax2.set_ylabel('t-SNE Component 2', fontsize=11)
    ax2.set_zlabel('t-SNE Component 3', fontsize=11)
    ax2.view_init(elev=20, azim=45)  # Same viewing angle
    
    # 공통 Legend
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=4, 
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='Anatomical Regions',
              title_fontsize=11)
    
    plt.suptitle(f'Feature Clustering {title_suffix} (3D t-SNE) - Full Volume', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save 3D
    save_path_3d = os.path.join(output_dir, f'patient_{patient_id}_tsne_3d_{suffix}.png')
    plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 3D: {save_path_3d}")
    
    # ========== Clustering Quality Metrics ==========
    from sklearn.metrics import silhouette_score
    
    if len(np.unique(baseline_labels)) > 1:
        # 2D Silhouette scores
        baseline_silhouette_2d = silhouette_score(baseline_2d, baseline_labels)
        proposed_silhouette_2d = silhouette_score(proposed_2d, proposed_labels)
        
        # 3D Silhouette scores
        baseline_silhouette_3d = silhouette_score(baseline_3d, baseline_labels)
        proposed_silhouette_3d = silhouette_score(proposed_3d, proposed_labels)
        
        print(f"\n  === 2D t-SNE Clustering Quality ===")
        print(f"  Silhouette Score (Baseline): {baseline_silhouette_2d:.4f}")
        print(f"  Silhouette Score (Proposed): {proposed_silhouette_2d:.4f}")
        print(f"  Improvement: {(proposed_silhouette_2d - baseline_silhouette_2d):.4f}")
        
        print(f"\n  === 3D t-SNE Clustering Quality ===")
        print(f"  Silhouette Score (Baseline): {baseline_silhouette_3d:.4f}")
        print(f"  Silhouette Score (Proposed): {proposed_silhouette_3d:.4f}")
        print(f"  Improvement: {(proposed_silhouette_3d - baseline_silhouette_3d):.4f}")
        
        return {
            'baseline_silhouette_2d': float(baseline_silhouette_2d),
            'proposed_silhouette_2d': float(proposed_silhouette_2d),
            'improvement_2d': float(proposed_silhouette_2d - baseline_silhouette_2d),
            'baseline_silhouette_3d': float(baseline_silhouette_3d),
            'proposed_silhouette_3d': float(proposed_silhouette_3d),
            'improvement_3d': float(proposed_silhouette_3d - baseline_silhouette_3d),
        }
    
    return {}


@click.command()
@click.option('--baseline_ckpt', type=str, required=True, help='Path to baseline model checkpoint')
@click.option('--proposed_ckpt', type=str, required=True, help='Path to proposed model checkpoint')
@click.option('--data_dir', type=str, default='data/imageCAS/test', help='Test data directory')
@click.option('--num_patients', type=int, default=5, help='Number of patients to visualize (used if --patient_ids not provided)')
@click.option('--patient_ids', type=str, default=None, help='Specific patient IDs to process (comma-separated, e.g., "880,980,1000")')
@click.option('--output_dir', type=str, default='result/experiments/feature_visualization_fullvolume', help='Output directory')
@click.option('--gpu_number', type=int, default=0, help='GPU number to use')
@click.option('--num_samples', type=int, default=1000, help='Number of samples per anatomical region for t-SNE')
@click.option('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
@click.option('--patch_size', type=int, default=96, help='Patch size for sliding window (e.g., 96 for 96x96x96)')
@click.option('--overlap', type=float, default=0.25, help='Overlap ratio for sliding window (0~1, default: 0.25 same as training)')
@click.option('--mode', type=str, default='gaussian', help='Weighting mode: "gaussian" or "constant" (default: "gaussian" same as training)')
def main(baseline_ckpt, proposed_ckpt, data_dir, num_patients, patient_ids, output_dir, gpu_number, num_samples, perplexity, patch_size, overlap, mode):
    """
    SPADE Feature Visualization - Full Volume Version
    
    96x96x96 패치로 전체 볼륨을 sliding window 인퍼런스하여 feature map을 추출하고,
    Gaussian weighted averaging으로 합친 후 해부학적 위치에 따른 feature clustering을 t-SNE로 분석합니다.
    
    Training과 동일한 설정: overlap=0.25, mode="gaussian"
    """
    print("="*70)
    print("SPADE FEATURE VISUALIZATION - FULL VOLUME")
    print("="*70)
    print(f"Baseline checkpoint: {baseline_ckpt}")
    print(f"Proposed checkpoint: {proposed_ckpt}")
    print(f"Data directory: {data_dir}")
    print(f"Patch size: {patch_size}x{patch_size}x{patch_size}")
    print(f"Overlap: {overlap}")
    print(f"Weighting mode: {mode}")
    
    # Determine patient list
    if patient_ids:
        # Use specific patient IDs
        patient_dirs = [pid.strip() for pid in patient_ids.split(',')]
        print(f"Specific patients: {', '.join(patient_dirs)}")
    else:
        # Use first N patients
        patient_dirs = sorted(os.listdir(data_dir))[:num_patients]
        print(f"Number of patients: {num_patients}")
    
    print(f"Output directory: {output_dir}")
    print(f"GPU: {gpu_number}")
    print("="*70)
    
    # Setup
    device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    baseline_model, proposed_model = load_models(baseline_ckpt, proposed_ckpt, device)
    
    # Get transforms
    baseline_transform, proposed_transform = get_transforms()
    
    # Store results
    all_results = []
    
    roi_size = (patch_size, patch_size, patch_size)
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir
        print(f"\n{'='*70}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'='*70}")
        
        # Prepare file paths
        img_path = os.path.join(data_dir, patient_dir, 'img.nii.gz')
        label_path = os.path.join(data_dir, patient_dir, 'label.nii.gz')
        seg_path = os.path.join(data_dir, patient_dir, 'heart_combined.nii.gz')
        
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            print(f"Skipping {patient_id} - missing files")
            continue
        
        # Load and transform data for baseline (no seg)
        baseline_data = {
            "image": img_path,
            "label": label_path,
        }
        baseline_data = baseline_transform(baseline_data)
        baseline_image = baseline_data["image"].unsqueeze(0)  # Keep on CPU for now
        baseline_label = baseline_data["label"].unsqueeze(0)
        
        # Load and transform data for proposed (with seg -> distance map)
        proposed_data = {
            "image": img_path,
            "label": label_path,
            "seg": seg_path,
        }
        proposed_data = proposed_transform(proposed_data)
        proposed_image = proposed_data["image"].unsqueeze(0)  # Keep on CPU for now
        proposed_label = proposed_data["label"].unsqueeze(0)
        proposed_segmap = proposed_data["seg"].unsqueeze(0)  # [1, 8, D, H, W]
        
        # Extract features from full volume using sliding window with Gaussian weighted averaging
        print("\nExtracting features from baseline model (full volume with sliding window + Gaussian weighting)...")
        baseline_features = extract_features_from_patient_fullvolume(
            baseline_model, baseline_image, None, device, patch_size=roi_size, overlap=overlap, use_segmap=False, mode=mode
        )
        
        print("\nExtracting features from proposed model (full volume with sliding window + Gaussian weighting)...")
        proposed_features = extract_features_from_patient_fullvolume(
            proposed_model, proposed_image, proposed_segmap, device, patch_size=roi_size, overlap=overlap, use_segmap=True, mode=mode
        )
        
        # Load original segmentation for visualization (0-7 labels)
        seg_original = torch.from_numpy(nib.load(seg_path).get_fdata()).float().unsqueeze(0).unsqueeze(0)
        
        # Visualize feature maps (now full volume!)
        patient_output_dir = os.path.join(output_dir, f'patient_{patient_id}')
        os.makedirs(patient_output_dir, exist_ok=True)
        
        visualize_feature_maps(baseline_features, proposed_features, patient_output_dir, patient_id,
                              ct_image=proposed_image, segmap=proposed_segmap, label=proposed_label, 
                              seg_original=seg_original)
        
        # ========== t-SNE with Heart Segmentation (7 anatomical regions, no background) ==========
        print(f"\n[1/9] Sampling features by Heart Anatomical Regions (All 7 labels) from full volume...")
        print(f"  Segmentation shape: {seg_original.shape}")
        baseline_sampled_heart, baseline_labels_heart = sample_features_by_anatomy(
            baseline_features, seg_original, num_samples=num_samples, exclude_background=True
        )
        proposed_sampled_heart, proposed_labels_heart = sample_features_by_anatomy(
            proposed_features, seg_original, num_samples=num_samples, exclude_background=True
        )
        
        # t-SNE visualization (Heart - All labels)
        print("Computing t-SNE for Heart Segmentation (All)...")
        tsne_results_heart = visualize_tsne(
            baseline_sampled_heart, proposed_sampled_heart,
            baseline_labels_heart, proposed_labels_heart,
            patient_output_dir, patient_id,
            perplexity=perplexity,
            suffix="heart_all"
        )
        
        # ========== t-SNE for Each Heart Label (Binary: target label vs others) ==========
        label_names = {
            1: 'coronary_arteries',
            2: 'aorta',
            3: 'myocardium',
            4: 'left_ventricle',
            5: 'right_ventricle',
            6: 'left_atrium',
            7: 'right_atrium'
        }
        
        tsne_results_individual = {}
        for target_label, label_name in label_names.items():
            print(f"\n[{target_label + 1}/9] Processing Heart Label {target_label}: {label_name}...")
            
            # Sample features for binary classification (target vs others)
            baseline_sampled_bin, baseline_labels_bin = sample_features_by_anatomy_binary(
                baseline_features, seg_original, target_label=target_label, num_samples=num_samples
            )
            proposed_sampled_bin, proposed_labels_bin = sample_features_by_anatomy_binary(
                proposed_features, seg_original, target_label=target_label, num_samples=num_samples
            )
            
            # t-SNE visualization (Binary)
            tsne_result_bin = visualize_tsne(
                baseline_sampled_bin, proposed_sampled_bin,
                baseline_labels_bin, proposed_labels_bin,
                patient_output_dir, patient_id,
                perplexity=perplexity,
                suffix=f"heart_{label_name}"
            )
            
            tsne_results_individual[f'heart_{label_name}'] = tsne_result_bin
        
        # Combine all results
        tsne_results = {
            **{f'heart_all_{k}': v for k, v in tsne_results_heart.items()},
            **{f'{label}_{k}': v for label, result in tsne_results_individual.items() for k, v in result.items()}
        }
        
        # Store results
        result = {
            'patient_id': patient_id,
            **tsne_results
        }
        all_results.append(result)
    
    # Save summary
    if all_results:
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        # Heart Segmentation t-SNE
        if 'heart_all_baseline_silhouette_2d' in all_results[0]:
            print(f"\n{'='*70}")
            print("HEART ANATOMICAL REGIONS (All 7 classes)")
            print(f"{'='*70}")
            
            # 2D metrics
            avg_baseline_2d = np.mean([r['heart_all_baseline_silhouette_2d'] for r in all_results])
            avg_proposed_2d = np.mean([r['heart_all_proposed_silhouette_2d'] for r in all_results])
            avg_improvement_2d = np.mean([r['heart_all_improvement_2d'] for r in all_results])
            
            # 3D metrics
            avg_baseline_3d = np.mean([r['heart_all_baseline_silhouette_3d'] for r in all_results])
            avg_proposed_3d = np.mean([r['heart_all_proposed_silhouette_3d'] for r in all_results])
            avg_improvement_3d = np.mean([r['heart_all_improvement_3d'] for r in all_results])
            
            print(f"\n=== 2D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_2d:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_2d:.4f}")
            print(f"Average Improvement: {avg_improvement_2d:.4f}")
            
            print(f"\n=== 3D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_3d:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_3d:.4f}")
            print(f"Average Improvement: {avg_improvement_3d:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
