"""
SPADE Feature Visualization
Decoder의 각 layer에서 SPADE 적용 여부에 따른 feature map 시각화 및 t-SNE 분석
"""

import autorootcwd
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import click
import os
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

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


# ========== 색상 설정 (모듈 레벨) ==========
# 각 해부학적 영역(label)에 대한 색상 지정 (HEX 코드 또는 matplotlib 색상 이름)
# Label 1-7에 해당하는 색상을 순서대로 지정

LABEL_COLORS = {
    1: '#f5c643',  # Coronary Arteries - 노란색
    2: '#d52e23',  # Aorta - 빨간색
    3: '#5ec64f',  # Myocardium - 녹색
    4: '#ec8675',  # Left Ventricle - 분홍색
    5: '#61d3fa',  # Right Ventricle - 파란색
    6: '#8133da',  # Left Atrium - 보라색
    7: '#173df5',  # Right Atrium - 파란색
}


# Binary classification의 경우 (label 0, 1)
BINARY_COLORS = {
    0: '#7f7f7f',  # Others - 회색
    1: '#1f77b4',  # Target - 파란색
}

# ========== Scatter Plot 설정 (모듈 레벨) ==========
SCATTER_SIZE = 12  # 원 크기
SCATTER_ALPHA = 0.6  # 투명도
EDGE_LINEWIDTH = 0.3  # 테두리 두께
EDGE_DARKNESS = 0.7  # 테두리 어두움 정도 (RGB 값에 곱할 값)

# ========== 제목 설정 (모듈 레벨) ==========
BASELINE_TITLE = '(a) Baseline (w/o AGC)'  # Baseline 모델 제목
PROPOSED_TITLE = '(b) Ours (w/ AGC)'  # Proposed 모델 제목
TITLE_FONTSIZE = 18  # 제목 글씨 크기


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


def extract_center_patch(volume, patch_size=(128, 128, 128)):
    """전체 볼륨에서 중앙 패치 추출
    
    Args:
        volume: [B, C, D, H, W] tensor
        patch_size: (d, h, w) tuple
        
    Returns:
        center_patch: [B, C, d, h, w] tensor
        crop_coords: (d_start, d_end, h_start, h_end, w_start, w_end) - crop 좌표
    """
    _, _, D, H, W = volume.shape
    d, h, w = patch_size
    
    # 중앙 좌표 계산
    d_start = max(0, (D - d) // 2)
    h_start = max(0, (H - h) // 2)
    w_start = max(0, (W - w) // 2)
    
    d_end = min(D, d_start + d)
    h_end = min(H, h_start + h)
    w_end = min(W, w_start + w)
    
    # 패치 추출
    center_patch = volume[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
    
    crop_coords = (d_start, d_end, h_start, h_end, w_start, w_end)
    
    return center_patch, crop_coords


def extract_features_from_patient(model, image, segmap, device, patch_size=(128, 128, 128), use_segmap=False):
    """환자 데이터에서 feature 추출 (중앙 패치만 사용)
    
    Returns:
        features: dict of extracted features
        crop_coords: tuple of (d_start, d_end, h_start, h_end, w_start, w_end)
    """
    extractor = FeatureExtractor()
    
    # Decoder의 up_layers에 hook 등록
    layer_names = ['up_layers.0', 'up_layers.1', 'up_layers.2']
    extractor.register_hooks(model, layer_names)
    
    # 중앙 패치 추출
    center_image, crop_coords = extract_center_patch(image, patch_size)
    
    # Inference on center patch only
    with torch.no_grad():
        if use_segmap:
            # Proposed model: segmap 사용
            center_segmap, _ = extract_center_patch(segmap, patch_size)
            _ = model(center_image, center_segmap)
        else:
            # Baseline model: segmap 미사용
            _ = model(center_image)
    
    features = extractor.features.copy()
    extractor.remove_hooks()
    
    return features, crop_coords


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
        # mid_slice = 80
        
        # 각 feature map의 평균 (채널 평균)
        baseline_slice = baseline_feat[:, :, :, mid_slice].mean(dim=0).numpy()
        proposed_slice = proposed_feat[:, :, :, mid_slice].mean(dim=0).numpy()
        
        # 차이 계산
        diff = proposed_slice - baseline_slice
        
        # CT image와 segmentation을 feature map 크기에 맞게 resize
        if ct_image is not None and segmap is not None:
            from scipy.ndimage import zoom
            
            # CT slice
            ct_mid_slice = ct_image.shape[4] // 2
            # ct_mid_slice = 80
            ct_slice = ct_image[0, 0, :, :, ct_mid_slice].cpu().numpy()
            
            # Segmentation slice (use original labels, not distance map!)
            if seg_original is not None:
                seg_mid_slice = seg_original.shape[4] // 2
                # seg_mid_slice = 80
                seg_slice = seg_original[0, 0, :, :, seg_mid_slice].cpu().numpy()
            else:
                # Fallback: use argmin on distance map (closest structure)
                seg_mid_slice = segmap.shape[4] // 2
                seg_slice = torch.argmin(torch.abs(segmap[0, :, :, :, seg_mid_slice]), dim=0).cpu().numpy()
            
            # Label slice (coronary artery ground truth)
            if label is not None:
                label_mid_slice = label.shape[4] // 2
                # label_mid_slice = 80
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
            print(f"mid_slice: {mid_slice}, ct_mid_slice: {ct_mid_slice}, seg_mid_slice: {seg_mid_slice}, label_mid_slice: {label_mid_slice}")
            
            # ========== Row 1: Anatomical Context ==========
            # CT image (resize to feature map size for consistent display)
            ct_slice_resized = zoom(ct_slice, zoom_factors, order=1)
            axes[0, 0].imshow(ct_slice_resized, cmap='gray', interpolation='bilinear')
            axes[0, 0].set_title(f'CT Image\nMid-slice {ct_mid_slice} (resized to feature size)', fontsize=11, )
            axes[0, 0].axis('off')
            
            # Heart segmentation overlay on CT (resize to feature map size)
            seg_slice_resized = zoom(seg_slice, zoom_factors, order=0)  # nearest neighbor for labels
            axes[0, 1].imshow(ct_slice_resized, cmap='gray', interpolation='bilinear')
            seg_overlay_resized = np.ma.masked_where(seg_slice_resized == 0, seg_slice_resized)
            axes[0, 1].imshow(seg_overlay_resized, cmap=region_cmap, alpha=0.5, vmin=0, vmax=7, interpolation='nearest')
            axes[0, 1].set_title(f'Heart Segmentation\n(8 Anatomical Regions)', fontsize=11, )
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
                axes[0, 2].set_title(f'Coronary Artery Label\n(Ground Truth, resized)', fontsize=11, )
            else:
                axes[0, 2].text(0.5, 0.5, 'Label Not Available', ha='center', va='center', 
                               fontsize=12, transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
            
            # ========== Row 2: Feature Maps ==========
            # Baseline
            im1 = axes[1, 0].imshow(baseline_slice, cmap='viridis')
            axes[1, 0].set_title(f'Baseline Features\n{layer_name}', fontsize=11, )
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
            
            # Proposed
            im2 = axes[1, 1].imshow(proposed_slice, cmap='viridis')
            axes[1, 1].set_title(f'Proposed (SPADE) Features\n{layer_name}', fontsize=11, )
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
            
            # Difference (Proposed - Baseline)
            im3 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[1, 2].set_title(f'Difference\n(Proposed - Baseline)', fontsize=11, )
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
            
            # ========== Row 3: Overlays and Analysis ==========
            # Absolute difference
            im4 = axes[2, 0].imshow(np.abs(diff), cmap='hot')
            axes[2, 0].set_title(f'Absolute Difference', fontsize=11, )
            axes[2, 0].axis('off')
            plt.colorbar(im4, ax=axes[2, 0], fraction=0.046)
            
            # Overlay: Absolute difference on CT
            axes[2, 1].imshow(ct_resized, cmap='gray', alpha=0.7)
            diff_overlay = np.ma.masked_where(np.abs(diff) < np.percentile(np.abs(diff), 50), np.abs(diff))
            im5 = axes[2, 1].imshow(diff_overlay, cmap='hot', alpha=0.8)
            axes[2, 1].set_title(f'Feature Change Overlay\n(on CT)', fontsize=11, )
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
                axes[2, 2].set_title(f'Feature Change\n(with Coronary Context)', fontsize=11, )
            else:
                axes[2, 2].imshow(ct_resized, cmap='gray', alpha=0.7)
                seg_overlay_resized = np.ma.masked_where(seg_slice_resized == 0, seg_slice_resized)
                axes[2, 2].imshow(seg_overlay_resized, cmap=region_cmap, alpha=0.3, vmin=0, vmax=7, interpolation='nearest')
                axes[2, 2].imshow(diff_overlay, cmap='hot', alpha=0.6)
                axes[2, 2].set_title(f'Feature Change\n(with Anatomical Context)', fontsize=11, )
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
        
        plt.suptitle(f'Feature Maps Comparison - Patient {patient_id}', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top for suptitle
        
        # Save
        safe_layer_name = layer_name.replace('.', '_')
        save_path = os.path.join(output_dir, f'patient_{patient_id}_layer_{safe_layer_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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


def save_tsne_results(baseline_2d, baseline_3d, proposed_2d, proposed_3d, 
                      baseline_labels, proposed_labels, output_dir, patient_id, suffix):
    """t-SNE 결과를 CSV로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline results
    baseline_df = pd.DataFrame({
        'patient_id': patient_id,
        'model': 'baseline',
        'label': baseline_labels,
        'tsne_2d_x': baseline_2d[:, 0],
        'tsne_2d_y': baseline_2d[:, 1],
        'tsne_3d_x': baseline_3d[:, 0],
        'tsne_3d_y': baseline_3d[:, 1],
        'tsne_3d_z': baseline_3d[:, 2],
    })
    
    # Proposed results
    proposed_df = pd.DataFrame({
        'patient_id': patient_id,
        'model': 'proposed',
        'label': proposed_labels,
        'tsne_2d_x': proposed_2d[:, 0],
        'tsne_2d_y': proposed_2d[:, 1],
        'tsne_3d_x': proposed_3d[:, 0],
        'tsne_3d_y': proposed_3d[:, 1],
        'tsne_3d_z': proposed_3d[:, 2],
    })
    
    # Combine
    combined_df = pd.concat([baseline_df, proposed_df], ignore_index=True)
    
    # Save
    csv_path = os.path.join(output_dir, f'patient_{patient_id}_tsne_{suffix}.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"  Saved t-SNE results to: {csv_path}")
    
    return csv_path


def load_tsne_results(csv_path):
    """CSV에서 t-SNE 결과 로드"""
    df = pd.read_csv(csv_path)
    
    # Baseline
    baseline_df = df[df['model'] == 'baseline']
    baseline_2d = baseline_df[['tsne_2d_x', 'tsne_2d_y']].values
    baseline_3d = baseline_df[['tsne_3d_x', 'tsne_3d_y', 'tsne_3d_z']].values
    baseline_labels = baseline_df['label'].values
    
    # Proposed
    proposed_df = df[df['model'] == 'proposed']
    proposed_2d = proposed_df[['tsne_2d_x', 'tsne_2d_y']].values
    proposed_3d = proposed_df[['tsne_3d_x', 'tsne_3d_y', 'tsne_3d_z']].values
    proposed_labels = proposed_df['label'].values
    
    patient_id = df['patient_id'].iloc[0]
    
    return baseline_2d, baseline_3d, proposed_2d, proposed_3d, baseline_labels, proposed_labels, patient_id


def visualize_tsne(baseline_features, proposed_features, baseline_labels, proposed_labels, 
                   output_dir, patient_id, perplexity=30, suffix="heart_all", save_cache=True):
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
    
    # Discrete colormap - 각 해부학적 영역별 색상 직접 지정
    unique_labels = np.unique(baseline_labels)
    
    # 모듈 레벨의 색상 딕셔너리 및 scatter plot 설정 사용
    # (LABEL_COLORS, BINARY_COLORS, SCATTER_SIZE, SCATTER_ALPHA, EDGE_LINEWIDTH, EDGE_DARKNESS)
    
    # ========== Calculate Silhouette Scores for 2D ==========
    from sklearn.metrics import silhouette_score
    baseline_silhouette_2d = None
    proposed_silhouette_2d = None
    baseline_silhouette_3d = None
    proposed_silhouette_3d = None
    if len(np.unique(baseline_labels)) > 1:
        baseline_silhouette_2d = silhouette_score(baseline_2d, baseline_labels)
        proposed_silhouette_2d = silhouette_score(proposed_2d, proposed_labels)
        baseline_silhouette_3d = silhouette_score(baseline_3d, baseline_labels)
        proposed_silhouette_3d = silhouette_score(proposed_3d, proposed_labels)
    
    # ========== 2D Visualization ==========
    print("  Creating 2D visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 각 subplot의 가로 길이 증가
    
    # Scatter plot alpha 설정: heart_all이 아닌 경우 0.9
    scatter_alpha = SCATTER_ALPHA if suffix == "heart_all" else 0.9
    
    # Baseline 2D
    for label in unique_labels:
        mask = baseline_labels == label

        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        axes[0].scatter(
            baseline_2d[mask, 0], baseline_2d[mask, 1],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    axes[0].set_title(BASELINE_TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    axes[0].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[0].tick_params(axis='both', which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0].set_facecolor('#eaeaf1')  # 푸른끼 도는 그레이
    axes[0].grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
    # 실루엣 스코어 표시
    if baseline_silhouette_2d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        axes[0].text(0.98, 0.96, f'Silhouette Score: {baseline_silhouette_2d:.2f}', 
                    transform=axes[0].transAxes, fontsize=14, color='#1c222b',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # Proposed 2D
    legend_handles = []
    legend_labels = []
    for label in unique_labels:
        mask = proposed_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        h = axes[1].scatter(
            proposed_2d[mask, 0], proposed_2d[mask, 1],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,  # 어두운 버전의 테두리
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
        legend_handles.append(h)
        legend_labels.append(f'{int(label)}: {label_names.get(int(label), "Unknown")}')
    
    axes[1].set_title(PROPOSED_TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    axes[1].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[1].tick_params(axis='both', which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[1].set_facecolor('#eaeaf1')
    axes[1].grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
    # 실루엣 스코어 표시
    if proposed_silhouette_2d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        axes[1].text(0.98, 0.96, f'Silhouette Score: {proposed_silhouette_2d:.2f}', 
                    transform=axes[1].transAxes, fontsize=14, color='#1c222b',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # 두 그래프의 xlim, ylim을 동일하게 설정하여 크기 일치
    xlim_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    xlim_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    ylim_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ylim_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    
    axes[0].set_xlim(xlim_min, xlim_max)
    axes[0].set_ylim(ylim_min, ylim_max)
    axes[1].set_xlim(xlim_min, xlim_max)
    axes[1].set_ylim(ylim_min, ylim_max)
    
    # 공통 Legend를 그래프 아래에 배치 (참고 이미지 스타일)
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),  # 그래프와 범례 사이 간격 좁힘
              ncol=7,  # 7개 항목을 한 줄로 배치
              fontsize=14,  # 범례 글씨 크기
              frameon=False,  # 테두리 제거
              columnspacing=1.5,
              handletextpad=0.5,
              scatterpoints=1,
              markerscale=2.5)  # 범례 마커 크기 더 증가
    
    # plt.suptitle(f'Feature Clustering {title_suffix} (2D t-SNE) - Patient {patient_id}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.subplots_adjust(wspace=0.5)  # 두 그래프 사이 간격 훨씬 더 띄우기
    
    # Save 2D
    save_path_2d = os.path.join(output_dir, f'patient_{patient_id}_tsne_2d_{suffix}.png')
    plt.savefig(save_path_2d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved 2D: {save_path_2d}")
    
    # ========== 3D Visualization ==========
    print("  Creating 3D visualization...")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 9))
    
    # Scatter plot alpha 설정: heart_all이 아닌 경우 0.9 (2D와 동일)
    scatter_alpha = SCATTER_ALPHA if suffix == "heart_all" else 0.9
    
    # Baseline 3D
    ax1 = fig.add_subplot(121, projection='3d')
    for label in unique_labels:
        mask = baseline_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        ax1.scatter(
            baseline_3d[mask, 0], baseline_3d[mask, 1], baseline_3d[mask, 2],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가 
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax1.set_title(BASELINE_TITLE, fontsize=TITLE_FONTSIZE, pad=20, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=13)
    ax1.set_ylabel('t-SNE Component 2', fontsize=13)
    ax1.set_zlabel('t-SNE Component 3', fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=20, azim=45)  # Set viewing angle
    # 실루엣 스코어 표시
    if baseline_silhouette_3d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        ax1.text2D(0.98, 0.96, f'Silhouette Score: {baseline_silhouette_3d:.2f}', 
                  transform=ax1.transAxes, fontsize=14, color='#1c222b',
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # Proposed 3D
    ax2 = fig.add_subplot(122, projection='3d')
    for label in unique_labels:
        mask = proposed_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        ax2.scatter(
            proposed_3d[mask, 0], proposed_3d[mask, 1], proposed_3d[mask, 2],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가 
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax2.set_title(PROPOSED_TITLE, fontsize=TITLE_FONTSIZE, pad=20, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=13)
    ax2.set_ylabel('t-SNE Component 2', fontsize=13)
    ax2.set_zlabel('t-SNE Component 3', fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.view_init(elev=20, azim=45)  # Same viewing angle
    # 실루엣 스코어 표시
    if proposed_silhouette_3d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        ax2.text2D(0.98, 0.96, f'Silhouette Score: {proposed_silhouette_3d:.2f}', 
                  transform=ax2.transAxes, fontsize=14, color='#1c222b',
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # 공통 Legend (참고 이미지 스타일)
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.03),  # 그래프와 범례 사이 간격 좁힘
              ncol=7,  # 7개 항목을 한 줄로 배치
              fontsize=14,  # 범례 글씨 크기
              frameon=False,  # 테두리 제거
              columnspacing=1.5,
              handletextpad=0.5,
              scatterpoints=1,
              markerscale=2.5)  # 범례 마커 크기 더 증가
    
    # plt.suptitle(f'Feature Clustering {title_suffix} (3D t-SNE)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save 3D
    save_path_3d = os.path.join(output_dir, f'patient_{patient_id}_tsne_3d_{suffix}.png')
    plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved 3D: {save_path_3d}")
    
    # ========== Save t-SNE Results to CSV ==========
    if save_cache:
        save_tsne_results(baseline_2d, baseline_3d, proposed_2d, proposed_3d,
                         baseline_labels, proposed_labels, output_dir, patient_id, suffix)
    
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


def visualize_tsne_from_cache(csv_path, output_dir, suffix="heart_all"):
    """
    저장된 CSV에서 t-SNE 결과를 로드하여 시각화만 수행
    (t-SNE 계산 없이 빠르게 그래프만 생성)
    """
    print(f"\nLoading t-SNE results from: {csv_path}")
    
    # Load data
    baseline_2d, baseline_3d, proposed_2d, proposed_3d, baseline_labels, proposed_labels, patient_id = load_tsne_results(csv_path)
    
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
        label_name_parts = suffix.replace("heart_", "").replace("_", " ").title()
        label_names = {
            0: 'Others',
            1: label_name_parts
        }
        title_suffix = f"{label_name_parts} vs Others"
    else:
        label_names = {0: 'Unknown'}
        title_suffix = suffix
    
    # Discrete colormap - 각 해부학적 영역별 색상 직접 지정
    unique_labels = np.unique(baseline_labels)
    
    # 모듈 레벨의 색상 딕셔너리 및 scatter plot 설정 사용
    # (LABEL_COLORS, BINARY_COLORS, SCATTER_SIZE, SCATTER_ALPHA, EDGE_LINEWIDTH, EDGE_DARKNESS)
    
    # ========== Calculate Silhouette Scores ==========
    from sklearn.metrics import silhouette_score
    baseline_silhouette_2d = None
    proposed_silhouette_2d = None
    baseline_silhouette_3d = None
    proposed_silhouette_3d = None
    if len(unique_labels) > 1:
        baseline_silhouette_2d = silhouette_score(baseline_2d, baseline_labels)
        proposed_silhouette_2d = silhouette_score(proposed_2d, proposed_labels)
        baseline_silhouette_3d = silhouette_score(baseline_3d, baseline_labels)
        proposed_silhouette_3d = silhouette_score(proposed_3d, proposed_labels)
    
    # ========== 2D Visualization ==========
    print("  Creating 2D visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 각 subplot의 가로 길이 증가
    
    # Scatter plot alpha 설정: heart_all이 아닌 경우 0.9
    scatter_alpha = SCATTER_ALPHA if suffix == "heart_all" else 0.9
    
    # Baseline 2D
    for label in unique_labels:
        mask = baseline_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        axes[0].scatter(
            baseline_2d[mask, 0], baseline_2d[mask, 1],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    axes[0].set_title(BASELINE_TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    axes[0].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[0].tick_params(axis='both', which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0].set_facecolor('#eaeaf1')  # 푸른끼 도는 그레이
    axes[0].grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
    # 실루엣 스코어 표시
    if baseline_silhouette_2d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        axes[0].text(0.98, 0.96, f'Silhouette Score: {baseline_silhouette_2d:.2f}', 
                    transform=axes[0].transAxes, fontsize=14, color='#1c222b',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # Proposed 2D
    legend_handles = []
    legend_labels = []
    for label in unique_labels:
        mask = proposed_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        h = axes[1].scatter(
            proposed_2d[mask, 0], proposed_2d[mask, 1],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,  # 어두운 버전의 테두리
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
        legend_handles.append(h)
        legend_labels.append(f'{int(label)}: {label_names.get(int(label), "Unknown")}')
    
    axes[1].set_title(PROPOSED_TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)
    axes[1].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[1].tick_params(axis='both', which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[1].set_facecolor('#eaeaf1')  # 푸른끼 도는 그레이
    axes[1].grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
    # 실루엣 스코어 표시
    if proposed_silhouette_2d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        axes[1].text(0.98, 0.96, f'Silhouette Score: {proposed_silhouette_2d:.2f}', 
                    transform=axes[1].transAxes, fontsize=14, color='#1c222b',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # 두 그래프의 xlim, ylim을 동일하게 설정하여 크기 일치
    xlim_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    xlim_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    ylim_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ylim_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    
    axes[0].set_xlim(xlim_min, xlim_max)
    axes[0].set_ylim(ylim_min, ylim_max)
    axes[1].set_xlim(xlim_min, xlim_max)
    axes[1].set_ylim(ylim_min, ylim_max)
    
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),  # 그래프와 범례 사이 간격 좁힘
              ncol=7,  # 7개 항목을 한 줄로 배치
              fontsize=14,  # 범례 글씨 크기
              frameon=False,  # 테두리 제거
              columnspacing=1.5,
              handletextpad=0.5,
              scatterpoints=1,
              markerscale=2.5)  # 범례 마커 크기 더 증가
    
    # plt.suptitle(f'Feature Clustering {title_suffix} (2D t-SNE)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.subplots_adjust(wspace=0.15)  # 두 그래프 사이 간격 조정
    
    save_path_2d = os.path.join(output_dir, f'patient_{patient_id}_tsne_2d_{suffix}.png')
    plt.savefig(save_path_2d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved 2D: {save_path_2d}")
    
    # ========== 3D Visualization ==========
    print("  Creating 3D visualization...")
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 9))
    
    # Scatter plot alpha 설정: heart_all이 아닌 경우 0.9 (2D와 동일)
    scatter_alpha = SCATTER_ALPHA if suffix == "heart_all" else 0.9
    
    # Baseline 3D
    ax1 = fig.add_subplot(121, projection='3d')
    for label in unique_labels:
        mask = baseline_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        ax1.scatter(
            baseline_3d[mask, 0], baseline_3d[mask, 1], baseline_3d[mask, 2],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax1.set_title(f'{BASELINE_TITLE}\nPatient {patient_id}', fontsize=TITLE_FONTSIZE, pad=20, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=13)
    ax1.set_ylabel('t-SNE Component 2', fontsize=13)
    ax1.set_zlabel('t-SNE Component 3', fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=20, azim=45)
    # 실루엣 스코어 표시
    if baseline_silhouette_3d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        ax1.text2D(0.98, 0.96, f'Silhouette Score: {baseline_silhouette_3d:.2f}', 
                  transform=ax1.transAxes, fontsize=14, color='#1c222b',
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    # Proposed 3D
    ax2 = fig.add_subplot(122, projection='3d')
    for label in unique_labels:
        mask = proposed_labels == label
        # Label에 직접 매핑된 색상 사용
        if suffix.startswith("heart_") and suffix != "heart_all":
            color = mcolors.to_rgba(BINARY_COLORS.get(int(label), '#000000'))
        else:
            color = mcolors.to_rgba(LABEL_COLORS.get(int(label), '#000000'))
        
        # 테두리를 더 어둡게 만들기 (RGB 값에 0.7을 곱함)
        edge_color = tuple(c * EDGE_DARKNESS if i < 3 else c for i, c in enumerate(color))
        
        ax2.scatter(
            proposed_3d[mask, 0], proposed_3d[mask, 1], proposed_3d[mask, 2],
            c=[color], s=SCATTER_SIZE, alpha=scatter_alpha,  # 원 크기 증가
            edgecolors=[edge_color], linewidths=EDGE_LINEWIDTH,
            label=f'{int(label)}: {label_names.get(int(label), "Unknown")}'
        )
    ax2.set_title(f'{PROPOSED_TITLE}\nPatient {patient_id}', fontsize=TITLE_FONTSIZE, pad=20, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=13)
    ax2.set_ylabel('t-SNE Component 2', fontsize=13)
    ax2.set_zlabel('t-SNE Component 3', fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.view_init(elev=20, azim=45)
    # 실루엣 스코어 표시
    if proposed_silhouette_3d is not None:
        alpha_score = 0.9  # 실루엣 스코어 배경 투명도 (모두 동일)
        ax2.text2D(0.98, 0.96, f'Silhouette Score: {proposed_silhouette_3d:.2f}', 
                  transform=ax2.transAxes, fontsize=14, color='#1c222b',
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=alpha_score, edgecolor='none'))
    
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.01),  # 그래프와 범례 사이 간격 좁힘
              ncol=4, 
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='Anatomical Regions',
              title_fontsize=11,
              markerscale=2.5)  # 범례 마커 크기 더 증가
    
    # plt.suptitle(f'Feature Clustering {title_suffix} (3D t-SNE)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    save_path_3d = os.path.join(output_dir, f'patient_{patient_id}_tsne_3d_{suffix}.png')
    plt.savefig(save_path_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved 3D: {save_path_3d}")
    
    # Clustering metrics
    from sklearn.metrics import silhouette_score
    
    if len(unique_labels) > 1:
        baseline_silhouette_2d = silhouette_score(baseline_2d, baseline_labels)
        proposed_silhouette_2d = silhouette_score(proposed_2d, proposed_labels)
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


@click.command()
@click.option('--baseline_ckpt', type=str, default=None, help='Path to baseline model checkpoint')
@click.option('--proposed_ckpt', type=str, default=None, help='Path to proposed model checkpoint')
@click.option('--data_dir', type=str, default='data/imageCAS/test', help='Test data directory')
@click.option('--num_patients', type=int, default=5, help='Number of patients to visualize (used if --patient_ids not provided)')
@click.option('--patient_ids', type=str, default=None, help='Specific patient IDs to process (comma-separated, e.g., "880,980,1000")')
@click.option('--output_dir', type=str, default='result/experiments/feature_visualization', help='Output directory')
@click.option('--gpu_number', type=int, default=0, help='GPU number to use')
@click.option('--num_samples', type=int, default=1000, help='Number of samples per anatomical region for t-SNE')
@click.option('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
@click.option('--load_cache', is_flag=True, help='Load t-SNE results from cached CSV files instead of computing')
@click.option('--cache_dir', type=str, default=None, help='Directory containing cached CSV files (default: same as output_dir)')
def main(baseline_ckpt, proposed_ckpt, data_dir, num_patients, patient_ids, output_dir, gpu_number, num_samples, perplexity, load_cache, cache_dir):
    """
    SPADE Feature Visualization
    
    Decoder의 각 layer에서 SPADE 적용 전후의 feature map을 시각화하고,
    해부학적 위치에 따른 feature clustering을 t-SNE로 분석합니다.
    
    --load_cache 옵션을 사용하면 저장된 CSV에서 t-SNE 결과를 불러와 빠르게 시각화할 수 있습니다.
    """
    print("="*70)
    print("SPADE FEATURE VISUALIZATION")
    print("="*70)
    
    # ========== Load from Cache Mode ==========
    if load_cache:
        print("MODE: Load from cached CSV files")
        cache_directory = cache_dir if cache_dir else output_dir
        print(f"Cache directory: {cache_directory}")
        print("="*70)
        
        # Determine patient list
        if patient_ids:
            patient_list = [pid.strip() for pid in patient_ids.split(',')]
        else:
            # Find all patient directories with cached CSV files
            patient_list = []
            for item in os.listdir(cache_directory):
                item_path = os.path.join(cache_directory, item)
                if os.path.isdir(item_path) and item.startswith('patient_'):
                    patient_list.append(item.replace('patient_', ''))
            patient_list = sorted(patient_list)[:num_patients]
        
        print(f"Patients to replot: {', '.join(patient_list)}")
        
        # Replot from cache
        for patient_id in tqdm(patient_list, desc="Replotting from cache"):
            patient_output_dir = os.path.join(cache_directory, f'patient_{patient_id}')
            
            if not os.path.exists(patient_output_dir):
                print(f"Skipping {patient_id} - directory not found")
                continue
            
            print(f"\n{'='*70}")
            print(f"Replotting Patient: {patient_id}")
            print(f"{'='*70}")
            
            # Find all CSV files for this patient
            csv_files = [f for f in os.listdir(patient_output_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"No CSV files found for patient {patient_id}")
                continue
            
            # Replot each cached result
            for csv_file in csv_files:
                csv_path = os.path.join(patient_output_dir, csv_file)
                suffix = csv_file.replace(f'patient_{patient_id}_tsne_', '').replace('.csv', '')
                
                print(f"\nReplotting: {suffix}")
                visualize_tsne_from_cache(csv_path, patient_output_dir, suffix=suffix)
        
        print(f"\n{'='*70}")
        print("REPLOTTING COMPLETE")
        print(f"{'='*70}")
        return
    
    # ========== Normal Mode (Compute t-SNE) ==========
    print("MODE: Compute t-SNE from features")
    if not baseline_ckpt or not proposed_ckpt:
        print("ERROR: --baseline_ckpt and --proposed_ckpt are required when not using --load_cache")
        return
    
    print(f"Baseline checkpoint: {baseline_ckpt}")
    print(f"Proposed checkpoint: {proposed_ckpt}")
    print(f"Data directory: {data_dir}")
    
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
        baseline_image = baseline_data["image"].unsqueeze(0).to(device)
        baseline_label = baseline_data["label"].unsqueeze(0).to(device)
        
        # Load and transform data for proposed (with seg -> distance map)
        proposed_data = {
            "image": img_path,
            "label": label_path,
            "seg": seg_path,
        }
        proposed_data = proposed_transform(proposed_data)
        proposed_image = proposed_data["image"].unsqueeze(0).to(device)
        proposed_label = proposed_data["label"].unsqueeze(0).to(device)
        proposed_segmap = proposed_data["seg"].unsqueeze(0).to(device)  # [1, 8, D, H, W]
        
        # Extract features from center patch
        print("Extracting features from baseline model (center patch)...")
        baseline_features, crop_coords = extract_features_from_patient(
            baseline_model, baseline_image, None, device, use_segmap=False
        )
        
        print("Extracting features from proposed model (center patch)...")
        proposed_features, _ = extract_features_from_patient(
            proposed_model, proposed_image, proposed_segmap, device, use_segmap=True
        )
        
        print(f"Center patch crop coordinates: D[{crop_coords[0]}:{crop_coords[1]}], H[{crop_coords[2]}:{crop_coords[3]}], W[{crop_coords[4]}:{crop_coords[5]}]")
        
        # Load original segmentation for visualization (0-7 labels)
        seg_original_full = torch.from_numpy(nib.load(seg_path).get_fdata()).float().unsqueeze(0).unsqueeze(0)
        
        # Extract same center patch from CT, segmap, label, seg_original for visualization
        d_start, d_end, h_start, h_end, w_start, w_end = crop_coords
        ct_patch = proposed_image[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
        segmap_patch = proposed_segmap[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
        label_patch = proposed_label[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
        seg_original_patch = seg_original_full[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
        
        # Visualize feature maps (now all spatially aligned!)
        patient_output_dir = os.path.join(output_dir, f'patient_{patient_id}')
        os.makedirs(patient_output_dir, exist_ok=True)
        
        visualize_feature_maps(baseline_features, proposed_features, patient_output_dir, patient_id,
                              ct_image=ct_patch, segmap=segmap_patch, label=label_patch, 
                              seg_original=seg_original_patch)
        
        # ========== t-SNE with Heart Segmentation (7 anatomical regions, no background) ==========
        # Use already cropped seg_original_patch
        seg_original = seg_original_patch
        
        print(f"\n[1/9] Sampling features by Heart Anatomical Regions (All 7 labels) from center patch...")
        print(f"  Segmentation patch shape: {seg_original.shape}")
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
            suffix="heart_all",
            save_cache=True
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
                suffix=f"heart_{label_name}",
                save_cache=True
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
        if 'heart_baseline_silhouette_2d' in all_results[0]:
            print(f"\n{'='*70}")
            print("HEART ANATOMICAL REGIONS (8 classes)")
            print(f"{'='*70}")
            
            # 2D metrics
            avg_baseline_2d = np.mean([r['heart_baseline_silhouette_2d'] for r in all_results])
            avg_proposed_2d = np.mean([r['heart_proposed_silhouette_2d'] for r in all_results])
            avg_improvement_2d = np.mean([r['heart_improvement_2d'] for r in all_results])
            
            # 3D metrics
            avg_baseline_3d = np.mean([r['heart_baseline_silhouette_3d'] for r in all_results])
            avg_proposed_3d = np.mean([r['heart_proposed_silhouette_3d'] for r in all_results])
            avg_improvement_3d = np.mean([r['heart_improvement_3d'] for r in all_results])
            
            print(f"\n=== 2D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_2d:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_2d:.4f}")
            print(f"Average Improvement: {avg_improvement_2d:.4f}")
            
            print(f"\n=== 3D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_3d:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_3d:.4f}")
            print(f"Average Improvement: {avg_improvement_3d:.4f}")
        
        # Coronary Artery Label t-SNE
        if 'coronary_baseline_silhouette_2d' in all_results[0]:
            print(f"\n{'='*70}")
            print("CORONARY ARTERY LABEL (Binary: 0/1)")
            print(f"{'='*70}")
            
            # 2D metrics
            avg_baseline_2d_cor = np.mean([r['coronary_baseline_silhouette_2d'] for r in all_results])
            avg_proposed_2d_cor = np.mean([r['coronary_proposed_silhouette_2d'] for r in all_results])
            avg_improvement_2d_cor = np.mean([r['coronary_improvement_2d'] for r in all_results])
            
            # 3D metrics
            avg_baseline_3d_cor = np.mean([r['coronary_baseline_silhouette_3d'] for r in all_results])
            avg_proposed_3d_cor = np.mean([r['coronary_proposed_silhouette_3d'] for r in all_results])
            avg_improvement_3d_cor = np.mean([r['coronary_improvement_3d'] for r in all_results])
            
            print(f"\n=== 2D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_2d_cor:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_2d_cor:.4f}")
            print(f"Average Improvement: {avg_improvement_2d_cor:.4f}")
            
            print(f"\n=== 3D t-SNE ===")
            print(f"Average Silhouette Score (Baseline): {avg_baseline_3d_cor:.4f}")
            print(f"Average Silhouette Score (Proposed): {avg_proposed_3d_cor:.4f}")
            print(f"Average Improvement: {avg_improvement_3d_cor:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
