"""
Visualize the effect of noise on conditioning (distance map)
Uses real test patient data with noisy dataloader
"""
import autorootcwd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from src.experiments.noisy_conditioning_dataloader import CoronaryArteryDataModuleNoisy


def visualize_noise_effect(
    data_dir="data/imageCAS",
    use_distance_map=True,
    noise_levels=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    seed=42,
    output_dir="result/experiments/noisy_conditioning"
):
    """
    실제 테스트 환자 데이터에서 노이즈 효과 시각화
    
    Args:
        data_dir: 데이터 디렉토리
        use_distance_map: Distance map 사용 여부
        noise_levels: 테스트할 노이즈 레벨들
        seed: 랜덤 시드
        output_dir: 출력 디렉토리
    """
    
    print("\n" + "="*70)
    print("실제 환자 데이터로 노이즈 효과 시각화")
    print("="*70)
    
    guide_type = "Distance Map" if use_distance_map else "Segmentation Map"
    print(f"Guide Type: {guide_type}")
    print(f"Noise Levels: {noise_levels}")
    print(f"Data Directory: {data_dir}")
    print("="*70 + "\n")
    
    # 1. 원본 데이터 로드 (노이즈 없음)
    print("📥 원본 환자 데이터 로딩 중...")
    data_module_original = CoronaryArteryDataModuleNoisy(
        data_dir=data_dir,
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=0,
        cache_rate=0.0,
        use_distance_map=use_distance_map,
        noise_std=0.0,
        noise_only_test=True,
        seed=seed
    )
    
    data_module_original.prepare_data()
    data_module_original.setup(stage="test")
    test_loader = data_module_original.test_dataloader()
    
    # 첫 번째 테스트 샘플 가져오기
    batch_original = next(iter(test_loader))
    
    # Get patient info
    filename = batch_original["image"].meta["filename_or_obj"][0]
    patient_id = filename.split("/")[-2]
    
    print(f"✅ Patient {patient_id} 데이터 로드 완료\n")
    
    # 원본 conditioning 추출
    seg_original = batch_original["seg"].cpu().numpy()  # [B, C, H, W, D]
    
    print(f"📐 Original conditioning shape: {seg_original.shape}")
    
    # 각 채널의 정보 출력
    channel_names = ["background", "coronary_arteries", "aorta", "myocardium", "ventricle_left", "ventricle_right", "atrium_left", "atrium_right"]
    num_channels = seg_original.shape[1]
    
    print(f"📐 각 채널 정보:")
    for ch_idx in range(num_channels):
        ch_data = seg_original[0, ch_idx]
        non_zero = np.sum(np.abs(ch_data) > 0.01)
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        print(f"  Channel {ch_idx} ({ch_name}): "
              f"{non_zero:,} non-zero pixels, "
              f"range [{ch_data.min():.2f}, {ch_data.max():.2f}]")
    
    # 2. 각 노이즈 레벨에 대해 데이터 로더로 conditioning 생성
    # (segmentation → noise → distance map 순서)
    print("\n🔄 노이즈 레벨별 conditioning 생성 중 (seg → noise → distance)...")
    
    # {noise_std: [C, H, W, D]} 형태로 저장
    conditioning_data = {}
    conditioning_data[0.0] = seg_original[0]  # [C, H, W, D] 원본 저장
    
    for noise_std in noise_levels[1:]:  # 0.0은 이미 저장했으므로 skip
        print(f"  🔄 Noise {noise_std} 데이터 로더 생성 중...")
        
        # 각 노이즈 레벨마다 새로운 데이터 모듈 생성
        data_module_noisy = CoronaryArteryDataModuleNoisy(
            data_dir=data_dir,
            batch_size=1,
            patch_size=(96, 96, 96),
            num_workers=0,
            cache_rate=0.0,
            use_distance_map=use_distance_map,
            noise_std=noise_std,
            noise_only_test=True,
            seed=seed
        )
        
        data_module_noisy.prepare_data()
        data_module_noisy.setup(stage="test")
        test_loader_noisy = data_module_noisy.test_dataloader()
        
        # 첫 번째 테스트 샘플 (같은 환자)
        batch_noisy = next(iter(test_loader_noisy))
        seg_noisy = batch_noisy["seg"].cpu().numpy()  # [B, C, H, W, D]
        
        conditioning_data[noise_std] = seg_noisy[0]
        print(f"    ✓ Noise {noise_std}: shape {seg_noisy[0].shape}, "
              f"range [{seg_noisy[0].min():.2f}, {seg_noisy[0].max():.2f}]")
    
    print("✅ 모든 노이즈 레벨 생성 완료 (seg → noise → distance map 순서)\n")
    
    # 3. 시각화할 슬라이스 선택 (중간 슬라이스)
    original = conditioning_data[0.0]  # [C, H, W, D]
    slice_idx = original.shape[3] // 2  # Z축(D) 중간
    
    print(f"📊 시각화 슬라이스: Z = {slice_idx}/{original.shape[3]} (shape: {original.shape})")
    
    # 4. 각 채널별로 시각화
    print("\n🎨 채널별 시각화 생성 중...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # vmin, vmax 설정 (distance map 기준)
    if use_distance_map:
        vmin, vmax = -50, 50
    else:
        vmin, vmax = 0, 7
    
    # 각 채널별로 시각화
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        
        # 해당 채널이 비어있으면 스킵
        if np.abs(original[ch_idx]).max() < 0.01:
            print(f"  ⊘ Channel {ch_idx} ({ch_name}): 비어있음, 스킵")
            continue
        
        print(f"  🖼️  Channel {ch_idx} ({ch_name}) 시각화 중...")
        
        fig, axes = plt.subplots(2, len(noise_levels), figsize=(4*len(noise_levels), 8))
        
        for idx, noise_std in enumerate(noise_levels):
            conditioning = conditioning_data[noise_std]  # [C, H, W, D]
            slice_data = conditioning[ch_idx, :, :, slice_idx]  # [H, W]
            
            # 상단: 2D 히트맵
            im = axes[0, idx].imshow(slice_data.T, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, idx].set_title(f'Noise σ = {noise_std}', fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
            
            # 하단: 중심 라인 프로파일
            center_x = slice_data.shape[0] // 2
            center_line = slice_data[center_x, :]
            
            axes[1, idx].plot(center_line, linewidth=2, label=f'σ={noise_std}')
            if use_distance_map:
                axes[1, idx].axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Boundary')
            axes[1, idx].set_ylim(vmin, vmax)
            axes[1, idx].set_xlabel('Position (voxels)', fontsize=10)
            axes[1, idx].set_ylabel('Distance (mm)' if use_distance_map else 'Label', fontsize=10)
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=axes[0, :], label='Distance (mm)' if use_distance_map else 'Label', 
                            fraction=0.046, pad=0.04)
        
        # 전체 타이틀
        fig.suptitle(f'Effect of Noise on {guide_type} - Patient {patient_id}\nChannel {ch_idx}: {ch_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 저장
        output_path = os.path.join(output_dir, f'noise_viz_patient_{patient_id}_ch{ch_idx}_{ch_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 저장: {output_path}")
        plt.close()
    
    print(f"\n✅ 모든 채널 시각화 완료")
    
    # 5. 채널별 통계 출력
    print("\n" + "="*70)
    print(f"📊 채널별 Conditioning 통계 - Patient {patient_id}")
    print("="*70)
    
    original = conditioning_data[0.0]  # [C, H, W, D]
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        ch_original = original[ch_idx]
        
        # 비어있는 채널 스킵
        if np.abs(ch_original).max() < 0.01:
            continue
        
        print(f"\n{'='*70}")
        print(f"Channel {ch_idx}: {ch_name}")
        print(f"{'='*70}")
        print(f"{'Noise Std':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'SNR':<12}")
        print("-"*70)
        
        original_std = ch_original.std()
        
        print(f"{'0.0 (orig)':<12} {ch_original.min():<12.2f} {ch_original.max():<12.2f} "
              f"{ch_original.mean():<12.2f} {ch_original.std():<12.2f} {'∞':<12}")
        
        for noise_std in noise_levels[1:]:
            noisy_ch = conditioning_data[noise_std][ch_idx]
            snr = original_std / noise_std if noise_std > 0 and original_std > 0 else float('inf')
            print(f"{noise_std:<12.1f} {noisy_ch.min():<12.2f} {noisy_ch.max():<12.2f} "
                  f"{noisy_ch.mean():<12.2f} {noisy_ch.std():<12.2f} {snr:<12.2f}")
        
        # 변화량 분석
        print(f"\n{'Noise Std':<12} {'Mean |Diff|':<15} {'Max |Diff|':<15} {'상태':<20}")
        print("-"*70)
        
        for noise_std in noise_levels[1:]:
            diff = conditioning_data[noise_std][ch_idx] - ch_original
            mean_abs_diff = np.abs(diff).mean()
            max_abs_diff = np.abs(diff).max()
            
            if mean_abs_diff < 0.1:
                status = "✅ 매우 작음"
            elif mean_abs_diff < 0.5:
                status = "🟢 작음"
            elif mean_abs_diff < 1.0:
                status = "🟡 중간"
            elif mean_abs_diff < 2.0:
                status = "🟠 큼"
            else:
                status = "🔴 매우 큼"
            
            print(f"{noise_std:<12.1f} {mean_abs_diff:<15.4f} {max_abs_diff:<15.4f} {status:<20}")
        
        # 경계 영역 분석 (distance map인 경우)
        if use_distance_map:
            boundary_mask = np.abs(ch_original) < 2.0
            boundary_pixels = boundary_mask.sum()
            
            if boundary_pixels > 0:
                print(f"\n🎯 경계 영역 분석 (|distance| < 2mm)")
                print(f"경계 픽셀 수: {boundary_pixels:,} ({100*boundary_pixels/ch_original.size:.2f}%)")
                print(f"{'Noise Std':<12} {'경계 Mean |Diff|':<20} {'2mm+ 변화 픽셀':<20}")
                print("-"*70)
                
                for noise_std in noise_levels[1:]:
                    noisy_ch = conditioning_data[noise_std][ch_idx]
                    
                    original_boundary = ch_original[boundary_mask]
                    noisy_boundary = noisy_ch[boundary_mask]
                    diff_boundary = noisy_boundary - original_boundary
                    
                    mean_abs_diff_boundary = np.abs(diff_boundary).mean()
                    large_shift = np.abs(diff_boundary) > 2.0
                    large_shift_pct = 100 * large_shift.sum() / boundary_pixels
                    
                    print(f"{noise_std:<12.1f} {mean_abs_diff_boundary:<20.4f} {large_shift_pct:<19.1f}%")
    
    print("\n" + "="*70)
    print("💡 해석")
    print("="*70)
    print("• 각 채널은 다른 장기의 distance map을 나타냄")
    print("• SNR (Signal-to-Noise Ratio) = signal_std / noise_std")
    print("• 높은 SNR = 깨끗한 신호, 노이즈 적음")
    print("• 낮은 SNR = 손상된 신호, 노이즈 많음")
    print("• Mean |Diff| < 0.5: 구조 잘 보존됨")
    print("• Mean |Diff| 0.5~1.0: 경계가 흐려지기 시작")
    print("• Mean |Diff| > 1.0: 구조가 크게 왜곡됨")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize noise effect on real patient data")
    parser.add_argument("--data_dir", type=str, default="data/imageCAS",
                        help="Data directory")
    parser.add_argument("--guide", type=str, default="distanceMap",
                        choices=["distanceMap", "segMap"],
                        help="Guide type")
    parser.add_argument("--noise_levels", type=str, default="0.0,0.5,1.0,2.0,5.0,10.0",
                        help="Comma-separated noise levels")
    parser.add_argument("--output_dir", type=str, 
                        default="result/experiments/noisy_conditioning",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Parse noise levels
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",")]
    
    visualize_noise_effect(
        data_dir=args.data_dir,
        use_distance_map=(args.guide == "distanceMap"),
        noise_levels=noise_levels,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n✅ 시각화 완료!")
