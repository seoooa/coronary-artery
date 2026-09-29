"""
Visualize the effect of SDM perturbations on distance maps

Two modes:
1. dilation_erosion: Add constant offset (over/under segmentation)
2. boundary_noise: Add Gaussian noise (boundary uncertainty)
"""
import autorootcwd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from src.experiments.sdm_perturbation_dataloader import CoronaryArteryDataModuleSDMPerturbation


def visualize_perturbation_effect(
    data_dir="data/imageCAS",
    perturbation_mode="boundary_noise",  # 'dilation_erosion' or 'boundary_noise'
    perturbation_levels=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    seed=42,
    output_dir="result/experiments/sdm_perturbation"
):
    """
    실제 테스트 환자 데이터에서 SDM perturbation 효과 시각화
    
    Args:
        data_dir: 데이터 디렉토리
        perturbation_mode: 'dilation_erosion' 또는 'boundary_noise'
        perturbation_levels: 테스트할 레벨들 (offset range 또는 noise std)
        seed: 랜덤 시드
        output_dir: 출력 디렉토리
    """
    
    print("\n" + "="*70)
    print("SDM Perturbation 효과 시각화")
    print("="*70)
    print(f"Perturbation Mode: {perturbation_mode}")
    print(f"Perturbation Levels: {perturbation_levels}")
    print(f"Data Directory: {data_dir}")
    print("="*70 + "\n")
    
    # 채널 이름
    channel_names = ["background", "coronary_arteries", "aorta", "myocardium", 
                     "ventricle_left", "ventricle_right", "atrium_left", "atrium_right"]
    
    # 1. 원본 데이터 로드 (perturbation 없음)
    print("📥 원본 환자 데이터 로딩 중...")
    data_module_original = CoronaryArteryDataModuleSDMPerturbation(
        data_dir=data_dir,
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=0,
        cache_rate=0.0,
        perturbation_mode=perturbation_mode,
        offset_range=0.0 if perturbation_mode == "dilation_erosion" else 2.0,
        noise_std=0.0 if perturbation_mode == "boundary_noise" else 0.5,
        perturbation_only_test=True,
        seed=seed
    )
    
    data_module_original.prepare_data()
    data_module_original.setup(stage="test")
    test_loader = data_module_original.test_dataloader()
    
    batch_original = next(iter(test_loader))
    
    filename = batch_original["image"].meta["filename_or_obj"][0]
    patient_id = filename.split("/")[-2]
    
    print(f"✅ Patient {patient_id} 데이터 로드 완료\n")
    
    sdm_original = batch_original["seg"].cpu().numpy()  # [B, C, H, W, D]
    
    print(f"📐 Original SDM shape: {sdm_original.shape}")
    print(f"📐 각 채널 정보:")
    
    num_channels = sdm_original.shape[1]
    
    for ch_idx in range(num_channels):
        ch_data = sdm_original[0, ch_idx]
        non_zero = np.sum(np.abs(ch_data) > 0.01)
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        print(f"  Channel {ch_idx} ({ch_name}): "
              f"{non_zero:,} non-zero pixels, "
              f"range [{ch_data.min():.2f}, {ch_data.max():.2f}]")
    
    # 2. 각 perturbation 레벨에 대해 SDM 생성
    print(f"\n🔄 Perturbation 레벨별 SDM 생성 중...")
    
    sdm_data = {}
    sdm_data[0.0] = sdm_original[0]  # [C, H, W, D]
    
    for level in perturbation_levels[1:]:
        print(f"  🔄 Level {level} 데이터 로더 생성 중...")
        
        data_module_perturbed = CoronaryArteryDataModuleSDMPerturbation(
            data_dir=data_dir,
            batch_size=1,
            patch_size=(96, 96, 96),
            num_workers=0,
            cache_rate=0.0,
            perturbation_mode=perturbation_mode,
            offset_range=level if perturbation_mode == "dilation_erosion" else 2.0,
            noise_std=level if perturbation_mode == "boundary_noise" else 0.5,
            perturbation_only_test=True,
            seed=seed
        )
        
        data_module_perturbed.prepare_data()
        data_module_perturbed.setup(stage="test")
        test_loader_perturbed = data_module_perturbed.test_dataloader()
        
        batch_perturbed = next(iter(test_loader_perturbed))
        sdm_perturbed = batch_perturbed["seg"].cpu().numpy()
        
        sdm_data[level] = sdm_perturbed[0]
        print(f"    ✓ Level {level}: shape {sdm_perturbed[0].shape}, "
              f"range [{sdm_perturbed[0].min():.2f}, {sdm_perturbed[0].max():.2f}]")
    
    print("✅ 모든 레벨 생성 완료\n")
    
    # 3. 시각화할 슬라이스 선택
    original = sdm_data[0.0]
    slice_idx = original.shape[3] // 2
    
    print(f"📊 시각화 슬라이스: Z = {slice_idx}/{original.shape[3]}")
    
    # 4. 각 채널별로 시각화
    print("\n🎨 채널별 시각화 생성 중...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    vmin, vmax = -30, 200
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        
        if np.abs(original[ch_idx]).max() < 0.01:
            print(f"  ⊘ Channel {ch_idx} ({ch_name}): 비어있음, 스킵")
            continue
        
        print(f"  🖼️  Channel {ch_idx} ({ch_name}) 시각화 중...")
        
        fig, axes = plt.subplots(2, len(perturbation_levels), figsize=(4*len(perturbation_levels), 8))
        
        for idx, level in enumerate(perturbation_levels):
            sdm = sdm_data[level]
            slice_data = sdm[ch_idx, :, :, slice_idx]
            
            # 상단: 2D 히트맵
            im = axes[0, idx].imshow(slice_data.T, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            if perturbation_mode == "dilation_erosion":
                axes[0, idx].set_title(f'Offset ±{level:.1f}mm', fontsize=12, fontweight='bold')
            else:
                axes[0, idx].set_title(f'Noise σ={level:.1f}mm', fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
            
            # 하단: 중심 라인 프로파일
            center_x = slice_data.shape[0] // 2
            center_line = slice_data[center_x, :]
            
            axes[1, idx].plot(center_line, linewidth=2, label=f'Level={level}')
            axes[1, idx].axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Boundary')
            axes[1, idx].set_ylim(vmin, vmax)
            axes[1, idx].set_xlabel('Position (voxels)', fontsize=10)
            axes[1, idx].set_ylabel('Distance (mm)', fontsize=10)
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=axes[0, :], label='Distance (mm)', fraction=0.046, pad=0.04)
        
        mode_title = "Dilation/Erosion" if perturbation_mode == "dilation_erosion" else "Boundary Noise"
        fig.suptitle(f'Effect of {mode_title} on SDM - Patient {patient_id}\nChannel {ch_idx}: {ch_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, perturbation_mode, 
                                    f'sdm_viz_{patient_id}_ch{ch_idx}_{ch_name}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 저장: {output_path}")
        plt.close()
    
    print(f"\n✅ 모든 채널 시각화 완료")
    
    # 5. 통계 출력
    print("\n" + "="*70)
    print(f"📊 채널별 SDM 통계 - Patient {patient_id}")
    print("="*70)
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        ch_original = original[ch_idx]
        
        if np.abs(ch_original).max() < 0.01:
            continue
        
        print(f"\n{'='*70}")
        print(f"Channel {ch_idx}: {ch_name}")
        print(f"{'='*70}")
        print(f"{'Level':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-"*70)
        
        print(f"{'0.0 (orig)':<12} {ch_original.min():<12.2f} {ch_original.max():<12.2f} "
              f"{ch_original.mean():<12.2f} {ch_original.std():<12.2f}")
        
        for level in perturbation_levels[1:]:
            perturbed_ch = sdm_data[level][ch_idx]
            print(f"{level:<12.1f} {perturbed_ch.min():<12.2f} {perturbed_ch.max():<12.2f} "
                  f"{perturbed_ch.mean():<12.2f} {perturbed_ch.std():<12.2f}")
        
        # 변화량 분석
        print(f"\n{'Level':<12} {'Mean |Diff|':<15} {'Max |Diff|':<15} {'상태':<20}")
        print("-"*70)
        
        for level in perturbation_levels[1:]:
            diff = sdm_data[level][ch_idx] - ch_original
            mean_abs_diff = np.abs(diff).mean()
            max_abs_diff = np.abs(diff).max()
            
            if mean_abs_diff < 0.5:
                status = "✅ 작음"
            elif mean_abs_diff < 1.0:
                status = "🟡 중간"
            elif mean_abs_diff < 2.0:
                status = "🟠 큼"
            else:
                status = "🔴 매우 큼"
            
            print(f"{level:<12.1f} {mean_abs_diff:<15.4f} {max_abs_diff:<15.4f} {status:<20}")
    
    print("\n" + "="*70)
    print("💡 해석")
    print("="*70)
    if perturbation_mode == "dilation_erosion":
        print("• 각 채널에 랜덤 상수 오프셋 추가")
        print("• 양수 오프셋 = 팽창 (Over-segmentation)")
        print("• 음수 오프셋 = 수축 (Under-segmentation)")
    else:
        print("• 가우시안 노이즈를 SDM에 직접 추가")
        print("• 경계 위치의 불확실성 시뮬레이션")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SDM perturbation effects")
    parser.add_argument("--data_dir", type=str, default="data/imageCAS",
                        help="Data directory")
    parser.add_argument("--mode", type=str, default="boundary_noise",
                        choices=["dilation_erosion", "boundary_noise"],
                        help="Perturbation mode")
    parser.add_argument("--levels", type=str, default="0.0,0.5,1.0,2.0,5.0,10.0",
                        help="Comma-separated perturbation levels")
    parser.add_argument("--output_dir", type=str, 
                        default="result/experiments/sdm_perturbation/visualization",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    perturbation_levels = [float(x.strip()) for x in args.levels.split(",")]
    
    visualize_perturbation_effect(
        data_dir=args.data_dir,
        perturbation_mode=args.mode,
        perturbation_levels=perturbation_levels,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n✅ 시각화 완료!")
