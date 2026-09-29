"""
Save segmentation map samples with seg_boundary_noise applied

This script saves original and noisy segmentation maps (before and after SDM conversion)
to verify that the seg_boundary_noise perturbation works correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
import torch
import os

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscreted,
    Compose,
)
from src.experiments.sdm_perturbation_dataloader import AddSegBoundaryNoised


def save_seg_samples(
    data_dir="data/imageCAS",
    patient_id="751",
    flip_probs=[0.0, 0.1, 0.3, 0.5],
    boundary_width=2.0,
    seed=42,
    output_dir="result/experiments/sdm_perturbation/seg_boundary_noise/samples"
):
    """
    Save segmentation map samples with different flip probabilities
    
    Args:
        data_dir: Data directory
        patient_id: Patient ID to visualize
        flip_probs: List of flip probabilities to test
        boundary_width: Boundary width in mm
        seed: Random seed
        output_dir: Output directory for saved files
    """
    
    print("\n" + "="*70)
    print("Saving Seg Boundary Noise Samples")
    print("="*70)
    print(f"Patient ID: {patient_id}")
    print(f"Flip Probabilities: {flip_probs}")
    print(f"Boundary Width: {boundary_width} mm")
    print(f"Output Directory: {output_dir}")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare file paths
    patient_dir = Path(data_dir) / "test" / patient_id
    seg_file = str(patient_dir / "heart_combined.nii.gz")
    
    if not os.path.exists(seg_file):
        print(f"❌ Error: Segmentation file not found: {seg_file}")
        return
    
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
    
    # Load and preprocess original segmentation
    print("📂 Loading original segmentation...")
    load_transforms = Compose([
        LoadImaged(keys=["seg"]),
        EnsureChannelFirstd(keys=["seg"]),
        Orientationd(keys=["seg"], axcodes="RAS"),
        AsDiscreted(keys=["seg"], to_onehot=8),
    ])
    
    data_dict = {"seg": seg_file}
    data_dict = load_transforms(data_dict)
    original_seg = data_dict["seg"]  # [C, H, W, D]
    
    # Get affine from loaded image
    affine = data_dict["seg"].meta["affine"].numpy()
    
    print(f"  ✓ Loaded: shape {original_seg.shape}, range [{original_seg.min():.2f}, {original_seg.max():.2f}]")
    
    # Save original segmentation
    print("\n💾 Saving original segmentation (flip_prob=0.0)...")
    original_label = torch.argmax(original_seg, dim=0).cpu().numpy().astype(np.uint8)
    original_label_file = os.path.join(output_dir, f"patient_{patient_id}_flip0.00_segmap.nii.gz")
    nib.save(nib.Nifti1Image(original_label, affine), original_label_file)
    print(f"  ✓ Saved: {original_label_file}")
    
    # Process each flip probability
    print(f"\n🔄 Processing {len(flip_probs)} flip probabilities...\n")
    
    for flip_prob in flip_probs:
        if flip_prob == 0.0:
            continue  # Already saved
        
        print(f"Processing flip_prob = {flip_prob}...")
        
        # Apply seg boundary noise
        noise_transform = AddSegBoundaryNoised(
            keys=["seg"],
            boundary_width=boundary_width,
            flip_prob=flip_prob,
            seed=seed
        )
        
        data_dict_noisy = {"seg": original_seg.clone()}
        data_dict_noisy = noise_transform(data_dict_noisy)
        noisy_seg = data_dict_noisy["seg"]  # [C, H, W, D] - Segmentation map with noise
        
        # Get perturbation info
        num_flipped = data_dict_noisy.get("num_flipped_pixels", 0)
        total_boundary = data_dict_noisy.get("total_boundary_pixels", 0)
        flip_ratio = num_flipped / total_boundary if total_boundary > 0 else 0
        
        print(f"  - Flipped pixels: {num_flipped}/{total_boundary} ({flip_ratio*100:.1f}%)")
        
        # Save noisy segmentation map
        noisy_label = torch.argmax(noisy_seg, dim=0).cpu().numpy().astype(np.uint8)
        noisy_segmap_file = os.path.join(output_dir, f"patient_{patient_id}_flip{flip_prob:.2f}_segmap.nii.gz")
        nib.save(nib.Nifti1Image(noisy_label, affine), noisy_segmap_file)
        print(f"  ✓ Saved: {noisy_segmap_file}")
        print()
    
    print("="*70)
    print("✅ All samples saved successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("="*70)
    
    # Print instructions
    print("\n📝 To view the results:")
    print(f"   cd {output_dir}")
    print(f"   # View with ITK-SNAP or other NIfTI viewer")
    print(f"\n   # Segmentation maps:")
    for flip_prob in flip_probs:
        print(f"   #   - patient_{patient_id}_flip{flip_prob:.2f}_segmap.nii.gz")
    print(f"\n   # Compare original (0.0) vs noisy versions to see the effect!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Save seg_boundary_noise samples")
    parser.add_argument("--data_dir", type=str, default="data/imageCAS", help="Data directory")
    parser.add_argument("--patient_id", type=str, default="751", help="Patient ID")
    parser.add_argument("--flip_probs", type=str, default="0.0,0.1,0.3,0.5", 
                        help="Comma-separated flip probabilities")
    parser.add_argument("--boundary_width", type=float, default=2.0, 
                        help="Boundary width in mm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, 
                        default="result/experiments/sdm_perturbation/seg_boundary_noise/samples",
                        help="Output directory")
    
    args = parser.parse_args()
    
    flip_probs = [float(x.strip()) for x in args.flip_probs.split(",")]
    
    save_seg_samples(
        data_dir=args.data_dir,
        patient_id=args.patient_id,
        flip_probs=flip_probs,
        boundary_width=args.boundary_width,
        seed=args.seed,
        output_dir=args.output_dir
    )
