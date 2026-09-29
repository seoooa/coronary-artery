"""
Robust Inference Script for Shifted Conditioning Experiment

This script evaluates pre-trained models with shifted conditioning (heart_combined.nii.gz)
to test robustness to spatial misalignment.

Based on proposed_train.py but focuses on inference with shifted conditioning.
"""

import autorootcwd
import lightning.pytorch as pytorch_lightning
from monai.transforms import (
    Compose,
    EnsureType,
    AsDiscrete,
)

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
import os
import click
import numpy as np
import nibabel as nib
from pathlib import Path
import csv

from src.experiments.shifted_conditioning_dataloader import CoronaryArteryDataModuleShifted
from script.proposed_train import CoronaryArterySegmentModel
from src.metrics.metrics import MetricFactory


def evaluate_with_shift(
    model,
    data_module,
    device,
    shift_range,
    output_dir,
    save_predictions=False
):
    """
    Evaluate model with specific shift range
    
    Args:
        model: Lightning model
        data_module: Data module with shifted conditioning
        device: Device to run on
        shift_range: Shift range in mm (physical distance)
        output_dir: Output directory for results
        save_predictions: Whether to save prediction volumes
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model = model.to(device)
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Initialize metrics
    metrics = MetricFactory.create_metrics()
    
    # Post-processing transforms
    post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
    
    # Storage for results
    all_results = []
    shift_info = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating with shift range: {shift_range} mm")
    print(f"{'='*70}")
    
    # Create output directory for this shift range
    shift_output_dir = Path(output_dir) / f"shift_{shift_range}"
    shift_output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            segs = batch["seg"].to(device)
            
            # Get patient ID
            filename = batch["image"].meta["filename_or_obj"][0]
            patient_id = filename.split("/")[-2]
            
            # Get shift amounts
            shift_amounts = batch.get("shift_amounts", (0, 0, 0))
            if torch.is_tensor(shift_amounts[0]):
                shift_x = shift_amounts[0].item()
                shift_y = shift_amounts[1].item()
                shift_z = shift_amounts[2].item()
            else:
                shift_x, shift_y, shift_z = shift_amounts
            
            # Get physical distance information
            shift_distance_mm = batch.get("shift_distance_mm", 0)
            shift_distance_voxels = batch.get("shift_distance_voxels", 0)
            voxel_spacing = batch.get("voxel_spacing", [0.35, 0.35, 0.5])
            
            if torch.is_tensor(shift_distance_mm):
                shift_distance_mm = shift_distance_mm.item()
            if torch.is_tensor(shift_distance_voxels):
                shift_distance_voxels = shift_distance_voxels.item()
            if torch.is_tensor(voxel_spacing):
                voxel_spacing = voxel_spacing.cpu().numpy()
            
            shift_info.append({
                "patient_id": patient_id,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "shift_z": shift_z,
                "shift_distance_mm": shift_distance_mm,
                "shift_distance_voxels": shift_distance_voxels,
            })
            
            print(f"Processing {patient_id} (shift: x={shift_x}, y={shift_y}, z={shift_z} voxels, distance={shift_distance_mm:.2f}mm)...")
            
            # Concatenate images and segs for sliding window inference
            inputs = torch.cat((images, segs), dim=1)
            
            # Sliding window inference
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            
            outputs = sliding_window_inference(
                inputs,
                roi_size,
                sw_batch_size,
                lambda x: model._model(x[:, :1, ...], x[:, 1:, ...]),
            )
            
            # Post-process
            outputs_processed = [post_pred(i) for i in decollate_batch(outputs)]
            labels_processed = [post_label(i) for i in decollate_batch(labels)]
            
            # Calculate metrics
            MetricFactory.calculate_metrics(metrics, outputs_processed, labels_processed)
            metric_results = MetricFactory.aggregate_metrics(metrics)
            MetricFactory.reset_metrics(metrics)
            
            # Store results
            result = {
                "patient_id": patient_id,
                "shift_range": shift_range,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "shift_z": shift_z,
                "shift_distance_mm": shift_distance_mm,
                "shift_distance_voxels": shift_distance_voxels,
                "dice": metric_results["dice"],
                "hausdorff": metric_results["hausdorff"],
                "iou": metric_results["iou"],
                "precision": metric_results["precision"],
                "recall": metric_results["recall"],
                "cldice": metric_results["cldice"],
                "betti_0": metric_results["betti_0"],
                "betti_1": metric_results["betti_1"],
            }
            all_results.append(result)
            
            print(f"  Dice: {metric_results['dice']:.4f}, "
                  f"Hausdorff: {metric_results['hausdorff']:.4f}, "
                  f"clDice: {metric_results['cldice']:.4f}")
            
            # Save predictions if requested
            if save_predictions:
                save_prediction(
                    images,
                    outputs_processed[0],
                    labels_processed[0],
                    shift_output_dir,
                    patient_id
                )
    
    # Calculate summary statistics
    dice_scores = [r["dice"] for r in all_results]
    hausdorff_scores = [r["hausdorff"] for r in all_results]
    iou_scores = [r["iou"] for r in all_results]
    precision_scores = [r["precision"] for r in all_results]
    recall_scores = [r["recall"] for r in all_results]
    cldice_scores = [r["cldice"] for r in all_results]
    betti_0_scores = [r["betti_0"] for r in all_results]
    betti_1_scores = [r["betti_1"] for r in all_results]
    
    summary = {
        "shift_range": shift_range,
        "mean_dice": np.mean(dice_scores),
        "std_dice": np.std(dice_scores),
        "mean_hausdorff": np.mean(hausdorff_scores),
        "std_hausdorff": np.std(hausdorff_scores),
        "mean_iou": np.mean(iou_scores),
        "std_iou": np.std(iou_scores),
        "mean_precision": np.mean(precision_scores),
        "std_precision": np.std(precision_scores),
        "mean_recall": np.mean(recall_scores),
        "std_recall": np.std(recall_scores),
        "mean_cldice": np.mean(cldice_scores),
        "std_cldice": np.std(cldice_scores),
        "mean_betti_0": np.mean(betti_0_scores),
        "std_betti_0": np.std(betti_0_scores),
        "mean_betti_1": np.mean(betti_1_scores),
        "std_betti_1": np.std(betti_1_scores),
        "num_samples": len(all_results),
    }
    
    # Save detailed results
    save_detailed_results(all_results, shift_output_dir, shift_range)
    
    return summary, all_results


def save_prediction(images, outputs, labels, output_dir, patient_id):
    """Save prediction volumes as NIfTI files"""
    
    images_np = images.detach().cpu().numpy().squeeze()
    outputs_np = outputs.detach().cpu().numpy().squeeze()[1]  # Get vessel class
    labels_np = labels.detach().cpu().numpy().squeeze()[1]
    
    affine = np.array([
        [0.35, 0, 0, 0],
        [0, 0.35, 0, 0],
        [0, 0, 0.5, 0],
        [0, 0, 0, 1]
    ])
    
    # Save input
    nib.save(
        nib.Nifti1Image(images_np, affine),
        output_dir / f"Subj_{patient_id}_input.nii.gz"
    )
    
    # Save prediction
    nib.save(
        nib.Nifti1Image(outputs_np, affine),
        output_dir / f"Subj_{patient_id}_prediction.nii.gz"
    )
    
    # Save ground truth
    nib.save(
        nib.Nifti1Image(labels_np, affine),
        output_dir / f"Subj_{patient_id}_label.nii.gz"
    )


def save_detailed_results(results, output_dir, shift_range):
    """Save detailed per-patient results with summary statistics"""
    
    result_file = output_dir / f"detailed_results_shift_{shift_range}.csv"
    
    # Calculate summary statistics
    dice_scores = [r["dice"] for r in results]
    hausdorff_scores = [r["hausdorff"] for r in results]
    iou_scores = [r["iou"] for r in results]
    precision_scores = [r["precision"] for r in results]
    recall_scores = [r["recall"] for r in results]
    cldice_scores = [r["cldice"] for r in results]
    betti_0_scores = [r["betti_0"] for r in results]
    betti_1_scores = [r["betti_1"] for r in results]
    shift_distance_mm_list = [r["shift_distance_mm"] for r in results]
    shift_distance_voxels_list = [r["shift_distance_voxels"] for r in results]
    
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "patient_id", "shift_range", "shift_x", "shift_y", "shift_z",
            "shift_distance_mm", "shift_distance_voxels",
            "dice", "hausdorff", "iou", "precision", "recall", 
            "cldice", "betti_0", "betti_1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write individual patient results
        for result in results:
            writer.writerow(result)
        
        # Write summary row (mean ± std)
        writer.writerow({
            "patient_id": "MEAN ± STD",
            "shift_range": shift_range,
            "shift_x": "-",
            "shift_y": "-",
            "shift_z": "-",
            "shift_distance_mm": f"{np.mean(shift_distance_mm_list):.2f} ± {np.std(shift_distance_mm_list):.2f}",
            "shift_distance_voxels": f"{np.mean(shift_distance_voxels_list):.2f} ± {np.std(shift_distance_voxels_list):.2f}",
            "dice": f"{np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}",
            "hausdorff": f"{np.mean(hausdorff_scores):.4f} ± {np.std(hausdorff_scores):.4f}",
            "iou": f"{np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}",
            "precision": f"{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}",
            "recall": f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}",
            "cldice": f"{np.mean(cldice_scores):.4f} ± {np.std(cldice_scores):.4f}",
            "betti_0": f"{np.mean(betti_0_scores):.4f} ± {np.std(betti_0_scores):.4f}",
            "betti_1": f"{np.mean(betti_1_scores):.4f} ± {np.std(betti_1_scores):.4f}",
        })
    
    print(f"Detailed results saved to: {result_file}")


def save_summary_results(summaries, output_dir, arch_name, guide):
    """Save summary results across all shift ranges"""
    
    guide_suffix = "dstMap" if guide == "distanceMap" else "segMap"
    result_file = output_dir / f"shift_{shift_range}/{arch_name}_{guide_suffix}_shift_robustness_summary.csv"
    
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "shift_range", "mean_dice", "std_dice", 
            "mean_hausdorff", "std_hausdorff",
            "mean_iou", "std_iou",
            "mean_precision", "std_precision",
            "mean_recall", "std_recall",
            "mean_cldice", "std_cldice",
            "mean_betti_0", "std_betti_0",
            "mean_betti_1", "std_betti_1",
            "num_samples"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for summary in summaries:
            writer.writerow({
                "shift_range": summary["shift_range"],
                "mean_dice": f"{summary['mean_dice']:.6f}",
                "std_dice": f"{summary['std_dice']:.6f}",
                "mean_hausdorff": f"{summary['mean_hausdorff']:.4f}",
                "std_hausdorff": f"{summary['std_hausdorff']:.4f}",
                "mean_iou": f"{summary['mean_iou']:.6f}",
                "std_iou": f"{summary['std_iou']:.6f}",
                "mean_precision": f"{summary['mean_precision']:.6f}",
                "std_precision": f"{summary['std_precision']:.6f}",
                "mean_recall": f"{summary['mean_recall']:.6f}",
                "std_recall": f"{summary['std_recall']:.6f}",
                "mean_cldice": f"{summary['mean_cldice']:.6f}",
                "std_cldice": f"{summary['std_cldice']:.6f}",
                "mean_betti_0": f"{summary['mean_betti_0']:.4f}",
                "std_betti_0": f"{summary['std_betti_0']:.4f}",
                "mean_betti_1": f"{summary['mean_betti_1']:.4f}",
                "std_betti_1": f"{summary['std_betti_1']:.4f}",
                "num_samples": summary["num_samples"],
            })
    
    print(f"\nSummary results saved to: {result_file}")
    return result_file


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(["SegResNet", "UNETR", "SwinUNETR", "nnFormer", "CSNet3D", "DSCNet", "AttentionUnet", "VNet"]),
    default="SegResNet",
    help="Architecture name",
)
@click.option(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to trained checkpoint (required)",
)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="distanceMap",
    help="Guide type used during training",
)
@click.option(
    "--data_dir",
    type=str,
    default="data/imageCAS",
    help="Data directory",
)
@click.option(
    "--shift_ranges",
    type=str,
    default="0,10,20,30",
    help="Comma-separated list of shift ranges to test (in mm, physical distance)",
)
@click.option(
    "--gpu_number",
    type=str,
    default="0",
    help="GPU number to use",
)
@click.option(
    "--output_dir",
    type=str,
    default="result/experiments/shifted_conditioning",
    help="Output directory for results",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--save_predictions",
    is_flag=True,
    default=True,
    help="Save prediction volumes as NIfTI files",
)
def main(
    arch_name,
    checkpoint_path,
    guide,
    data_dir,
    shift_ranges,
    gpu_number,
    output_dir,
    seed,
    save_predictions
):
    """
    Robust Inference - Evaluate model with shifted conditioning
    """
    
    print("\n" + "="*70)
    print("Shifted Conditioning Robustness Evaluation")
    print("="*70)
    print(f"Architecture: {arch_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Guide: {guide}")
    print(f"Data directory: {data_dir}")
    print(f"GPU: {gpu_number}")
    print(f"Random seed: {seed}")
    print(f"Save predictions: {save_predictions}")
    print("="*70 + "\n")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse shift ranges
    shift_range_list = [int(x.strip()) for x in shift_ranges.split(",")]
    print(f"Testing shift ranges: {shift_range_list}\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    print(f"Loading model from checkpoint...")
    model = CoronaryArterySegmentModel.load_from_checkpoint(
        checkpoint_path,
        arch_name=arch_name,
        loss_fn="DiceFocalLoss",  # Doesn't matter for inference
        batch_size=1
    )
    model.eval()
    print(f"Model loaded successfully!\n")
    
    # Storage for all results
    all_summaries = []
    
    # Evaluate for each shift range
    for shift_range in shift_range_list:
        print("\n" + "="*70)
        print(f"Shift Range: {shift_range} mm")
        print("="*70)
        
        # Initialize data module with specific shift range
        data_module = CoronaryArteryDataModuleShifted(
            data_dir=data_dir,
            batch_size=1,
            patch_size=(96, 96, 96),
            num_workers=4,
            cache_rate=0.0,  # No caching for testing
            use_distance_map=(guide == "distanceMap"),
            shift_range=shift_range,
            shift_only_test=True,
            seed=seed
        )
        
        data_module.prepare_data()
        data_module.setup(stage="test")
        
        # Evaluate
        summary, detailed_results = evaluate_with_shift(
            model,
            data_module,
            device,
            shift_range,
            output_path,
            save_predictions
        )
        
        all_summaries.append(summary)
        
        # Print summary for this shift range
        print(f"\nSummary for shift range {shift_range} mm:")
        print(f"  Dice:      {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"  Hausdorff: {summary['mean_hausdorff']:.4f} ± {summary['std_hausdorff']:.4f}")
        print(f"  IoU:       {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
        print(f"  Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
        print(f"  Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
        print(f"  clDice:    {summary['mean_cldice']:.4f} ± {summary['std_cldice']:.4f}")
        print(f"  Betti-0:   {summary['mean_betti_0']:.4f} ± {summary['std_betti_0']:.4f}")
        print(f"  Betti-1:   {summary['mean_betti_1']:.4f} ± {summary['std_betti_1']:.4f}")
    
    # Save summary across all shift ranges
    summary_file = save_summary_results(all_summaries, output_path, arch_name, guide)
    
    # Print final summary table
    print("\n\n" + "="*70)
    print("FINAL SUMMARY - Robustness to Shifted Conditioning")
    print("="*70)
    print(f"Architecture: {arch_name}")
    print(f"Guide Type: {guide}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\n")
    print(f"{'Shift':<12} {'Dice':<20} {'Hausdorff':<20} {'clDice':<20}")
    print("-" * 70)
    
    for summary in all_summaries:
        shift_str = f"{summary['shift_range']} mm"
        dice_str = f"{summary['mean_dice']:.4f}±{summary['std_dice']:.4f}"
        hausdorff_str = f"{summary['mean_hausdorff']:.2f}±{summary['std_hausdorff']:.2f}"
        cldice_str = f"{summary['mean_cldice']:.4f}±{summary['std_cldice']:.4f}"
        print(f"{shift_str:<12} {dice_str:<20} {hausdorff_str:<20} {cldice_str:<20}")
    
    # Calculate performance degradation
    if len(all_summaries) > 1:
        baseline_dice = all_summaries[0]["mean_dice"]
        baseline_cldice = all_summaries[0]["mean_cldice"]
        
        print("\n" + "="*70)
        print("Performance Degradation (relative to baseline)")
        print("="*70)
        print(f"{'Shift':<12} {'Dice Drop':<20} {'% Degradation':<20}")
        print("-" * 70)
        
        for summary in all_summaries[1:]:
            shift_str = f"{summary['shift_range']} mm"
            dice_drop = baseline_dice - summary['mean_dice']
            percent_degrade = (dice_drop / baseline_dice) * 100 if baseline_dice > 0 else 0
            dice_drop_str = f"{dice_drop:.4f}"
            percent_str = f"{percent_degrade:.2f}%"
            print(f"{shift_str:<12} {dice_drop_str:<20} {percent_str:<20}")
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Summary file: {summary_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
