"""
Visualize the effect of seg_boundary_noise on saved segmentation maps

This script:
1. Loads saved segmentation maps with different flip probabilities
2. Applies Signed Distance Transform to each
3. Visualizes the effect for each channel separately
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import os
from monai.transforms import AsDiscreted
from src.experiments.sdm_perturbation_dataloader import create_distance_map


def visualize_seg_boundary_noise(
    patient_id="751",
    flip_probs=[0.0, 0.1, 0.3, 0.5],
    input_dir="result/experiments/sdm_perturbation/seg_boundary_noise/samples",
    output_dir="result/experiments/sdm_perturbation/seg_boundary_noise/visualization",
    seed=42
):
    """
    Visualize seg_boundary_noise effect on each channel
    
    Args:
        patient_id: Patient ID
        flip_probs: List of flip probabilities
        input_dir: Directory containing saved segmentation maps
        output_dir: Output directory for visualizations
        seed: Random seed
    """
    
    print("\n" + "="*70)
    print("Visualizing Seg Boundary Noise Effect")
    print("="*70)
    print(f"Patient ID: {patient_id}")
    print(f"Flip Probabilities: {flip_probs}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Channel names
    channel_names = [
        "background",
        "coronary_arteries", 
        "aorta",
        "myocardium",
        "ventricle_left",
        "ventricle_right",
        "atrium_left",
        "atrium_right"
    ]
    num_channels = 8
    
    # Load segmentation maps and convert to distance maps
    print("🔄 Loading segmentation maps and converting to SDM...\n")
    
    sdm_data = {}  # {flip_prob: [C, H, W, D]}
    
    for flip_prob in flip_probs:
        print(f"  - Loading flip_prob = {flip_prob}")
        
        # Load segmentation map
        seg_file = os.path.join(input_dir, f"patient_{patient_id}_flip{flip_prob:.2f}_segmap.nii.gz")
        
        if not os.path.exists(seg_file):
            print(f"    ⚠️  File not found: {seg_file}")
            continue
        
        # Load NIfTI file
        nii = nib.load(seg_file)
        seg_label = nii.get_fdata().astype(np.uint8)  # [H, W, D]
        
        print(f"    Shape: {seg_label.shape}, unique labels: {np.unique(seg_label)}")
        
        # Convert to one-hot encoding
        seg_onehot = torch.zeros((num_channels, *seg_label.shape), dtype=torch.float32)
        for label in range(num_channels):
            seg_onehot[label] = torch.from_numpy((seg_label == label).astype(np.float32))
        
        # Convert to distance map
        sdm = create_distance_map(seg_onehot)
        
        sdm_data[flip_prob] = sdm.numpy()
        
        print(f"    ✓ SDM shape: {sdm.shape}, range: [{sdm.min():.2f}, {sdm.max():.2f}]")
    
    print("\n✅ All segmentation maps converted to SDM\n")
    
    # Select slice for visualization
    original = sdm_data[flip_probs[0]]
    slice_idx = original.shape[3] // 2
    print(f"📊 Visualization slice: Z = {slice_idx}/{original.shape[3]}\n")
    
    # Visualize each channel
    print("🎨 Creating visualizations for each channel...\n")
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        
        # Check if channel is empty
        if np.abs(original[ch_idx]).max() < 0.01:
            print(f"  ⊘ Channel {ch_idx} ({ch_name}): Empty, skipping")
            continue
        
        print(f"  🖼️  Channel {ch_idx} ({ch_name}) - Creating visualization...")
        
        # Create figure
        fig, axes = plt.subplots(2, len(flip_probs), figsize=(4*len(flip_probs), 8))
        
        # Set color limits based on data range
        vmin, vmax = -30, 200
        
        for idx, flip_prob in enumerate(flip_probs):
            if flip_prob not in sdm_data:
                continue
            
            sdm = sdm_data[flip_prob]
            slice_data = sdm[ch_idx, :, :, slice_idx]
            
            # Top: 2D heatmap
            im = axes[0, idx].imshow(slice_data.T, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, idx].set_title(f'Flip prob={flip_prob:.1f}', fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
            
            # Bottom: Center line profile
            center_x = slice_data.shape[0] // 2
            center_line = slice_data[center_x, :]
            
            axes[1, idx].plot(center_line, linewidth=2, label=f'Flip={flip_prob}')
            axes[1, idx].axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Boundary')
            axes[1, idx].set_ylim(vmin, vmax)
            axes[1, idx].set_xlabel('Position (voxels)', fontsize=10)
            axes[1, idx].set_ylabel('Distance (mm)', fontsize=10)
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, :], label='Distance (mm)', fraction=0.046, pad=0.04)
        
        # Add title
        fig.suptitle(f'Effect of Seg Boundary Noise on SDM - Patient {patient_id}\nChannel {ch_idx}: {ch_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'seg_noise_viz_{patient_id}_ch{ch_idx}_{ch_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("✅ All visualizations created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("="*70)
    
    # Print statistics
    print("\n📊 Statistics for each channel:\n")
    
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        
        if np.abs(original[ch_idx]).max() < 0.01:
            continue
        
        print(f"{'='*70}")
        print(f"Channel {ch_idx}: {ch_name}")
        print(f"{'='*70}")
        print(f"{'Flip Prob':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print(f"{'-'*70}")
        
        for flip_prob in flip_probs:
            if flip_prob not in sdm_data:
                continue
            
            sdm = sdm_data[flip_prob]
            ch_data = sdm[ch_idx]
            
            print(f"{flip_prob:<12.2f} {ch_data.min():<12.2f} {ch_data.max():<12.2f} "
                  f"{ch_data.mean():<12.2f} {ch_data.std():<12.2f}")
        
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize seg_boundary_noise effect")
    parser.add_argument("--patient_id", type=str, default="751", help="Patient ID")
    parser.add_argument("--flip_probs", type=str, default="0.0,0.1,0.3,0.5", 
                        help="Comma-separated flip probabilities")
    parser.add_argument("--input_dir", type=str, 
                        default="result/experiments/sdm_perturbation/seg_boundary_noise/samples",
                        help="Input directory with saved segmentation maps")
    parser.add_argument("--output_dir", type=str, 
                        default="result/experiments/sdm_perturbation/seg_boundary_noise/visualization",
                        help="Output directory for visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    flip_probs = [float(x.strip()) for x in args.flip_probs.split(",")]
    
    visualize_seg_boundary_noise(
        patient_id=args.patient_id,
        flip_probs=flip_probs,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
