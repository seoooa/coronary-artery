"""
Robust Inference Script for SDM Perturbation Experiment

This script evaluates pre-trained models with perturbed SDM (Signed Distance Map)
to test robustness to segmentation errors.

Two perturbation modes:
1. dilation_erosion: Add constant offset (over/under segmentation)
2. boundary_noise: Add Gaussian noise (boundary uncertainty)
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

from src.experiments.sdm_perturbation_dataloader import CoronaryArteryDataModuleSDMPerturbation
from script.proposed_train import CoronaryArterySegmentModel
from src.metrics.metrics import MetricFactory


def evaluate_with_perturbation(
    model,
    data_module,
    device,
    perturbation_mode,
    perturbation_level,
    output_dir,
    save_predictions=False
):
    """
    Evaluate model with specific perturbation level
    
    Args:
        model: Lightning model
        data_module: Data module with perturbed SDM
        device: Device to run on
        perturbation_mode: 'dilation_erosion' or 'boundary_noise'
        perturbation_level: Perturbation level (offset range or noise std)
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
    
    print(f"\n{'='*70}")
    print(f"Evaluating with {perturbation_mode}: level {perturbation_level}")
    print(f"{'='*70}")
    
    # Create output directory for this perturbation level
    level_output_dir = Path(output_dir) / perturbation_mode / f"level_{perturbation_level}"
    level_output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            segs = batch["seg"].to(device)
            
            # Get patient ID
            filename = batch["image"].meta["filename_or_obj"][0]
            patient_id = filename.split("/")[-2]
            
            # Get perturbation information
            perturbation_type = batch.get("perturbation_type", "none")
            
            if perturbation_mode == "dilation_erosion":
                channel_offsets = batch.get("channel_offsets", None)
                if channel_offsets is not None:
                    if torch.is_tensor(channel_offsets):
                        channel_offsets = channel_offsets.cpu().numpy()[0]
                    offsets_str = ", ".join([f"{x:.2f}" for x in channel_offsets])
                    print(f"Processing {patient_id} (channel offsets: [{offsets_str}])...")
                else:
                    print(f"Processing {patient_id}...")
            else:  # boundary_noise
                noise_std = batch.get("noise_std", 0.0)
                snr = batch.get("snr", float('inf'))
                if torch.is_tensor(noise_std):
                    noise_std = noise_std.item()
                if torch.is_tensor(snr):
                    snr = snr.item()
                print(f"Processing {patient_id} (noise_std: {noise_std:.2f}, SNR: {snr:.2f})...")
            
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
                "perturbation_mode": perturbation_mode,
                "perturbation_level": perturbation_level,
                "dice": metric_results["dice"],
                "hausdorff": metric_results["hausdorff"],
                "iou": metric_results["iou"],
                "precision": metric_results["precision"],
                "recall": metric_results["recall"],
                "cldice": metric_results["cldice"],
                "betti_0": metric_results["betti_0"],
                "betti_1": metric_results["betti_1"],
            }
            
            if perturbation_mode == "boundary_noise":
                result["noise_std"] = noise_std
                result["snr"] = snr
            
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
                    level_output_dir,
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
        "perturbation_mode": perturbation_mode,
        "perturbation_level": perturbation_level,
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
    save_detailed_results(all_results, level_output_dir, perturbation_mode, perturbation_level)
    
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


def save_detailed_results(results, output_dir, perturbation_mode, perturbation_level):
    """Save detailed per-patient results with summary statistics"""
    
    result_file = output_dir / f"detailed_results_{perturbation_mode}_{perturbation_level}.csv"
    
    # Calculate summary statistics
    dice_scores = [r["dice"] for r in results]
    hausdorff_scores = [r["hausdorff"] for r in results]
    iou_scores = [r["iou"] for r in results]
    precision_scores = [r["precision"] for r in results]
    recall_scores = [r["recall"] for r in results]
    cldice_scores = [r["cldice"] for r in results]
    betti_0_scores = [r["betti_0"] for r in results]
    betti_1_scores = [r["betti_1"] for r in results]
    
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "patient_id", "perturbation_mode", "perturbation_level",
            "dice", "hausdorff", "iou", "precision", "recall", 
            "cldice", "betti_0", "betti_1"
        ]
        
        if perturbation_mode == "boundary_noise":
            fieldnames.extend(["noise_std", "snr"])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write individual patient results
        for result in results:
            writer.writerow(result)
        
        # Write summary row (mean ± std)
        summary_row = {
            "patient_id": "MEAN ± STD",
            "perturbation_mode": perturbation_mode,
            "perturbation_level": perturbation_level,
            "dice": f"{np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}",
            "hausdorff": f"{np.mean(hausdorff_scores):.4f} ± {np.std(hausdorff_scores):.4f}",
            "iou": f"{np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}",
            "precision": f"{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}",
            "recall": f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}",
            "cldice": f"{np.mean(cldice_scores):.4f} ± {np.std(cldice_scores):.4f}",
            "betti_0": f"{np.mean(betti_0_scores):.4f} ± {np.std(betti_0_scores):.4f}",
            "betti_1": f"{np.mean(betti_1_scores):.4f} ± {np.std(betti_1_scores):.4f}",
        }
        
        if perturbation_mode == "boundary_noise":
            noise_stds = [r["noise_std"] for r in results]
            snrs = [r["snr"] for r in results if r["snr"] != float('inf')]
            summary_row["noise_std"] = f"{np.mean(noise_stds):.4f}"
            summary_row["snr"] = f"{np.mean(snrs):.2f}" if snrs else "∞"
        
        writer.writerow(summary_row)
    
    print(f"Detailed results saved to: {result_file}")


def save_summary_results(summaries, output_dir, arch_name, perturbation_mode):
    """Save summary results across all perturbation levels"""
    
    result_file = output_dir / perturbation_mode / f"{arch_name}_{perturbation_mode}_robustness_summary.csv"
    
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "perturbation_level", "mean_dice", "std_dice", 
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
                "perturbation_level": summary["perturbation_level"],
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
    "--data_dir",
    type=str,
    default="data/imageCAS",
    help="Data directory",
)
@click.option(
    "--perturbation_mode",
    type=click.Choice(["dilation_erosion", "boundary_noise"]),
    default="boundary_noise",
    help="Perturbation mode",
)
@click.option(
    "--perturbation_levels",
    type=str,
    default="0.0,0.5,1.0,2.0,5.0",
    help="Comma-separated list of perturbation levels to test",
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
    default="result/experiments/sdm_perturbation",
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
    data_dir,
    perturbation_mode,
    perturbation_levels,
    gpu_number,
    output_dir,
    seed,
    save_predictions
):
    """
    Robust Inference - Evaluate model with perturbed SDM
    """
    
    print("\n" + "="*70)
    print("SDM Perturbation Robustness Evaluation")
    print("="*70)
    print(f"Architecture: {arch_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Perturbation Mode: {perturbation_mode}")
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
    
    # Parse perturbation levels
    perturbation_level_list = [float(x.strip()) for x in perturbation_levels.split(",")]
    print(f"Testing {perturbation_mode} levels: {perturbation_level_list}\n")
    
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
    
    # Evaluate for each perturbation level
    for level in perturbation_level_list:
        print("\n" + "="*70)
        if perturbation_mode == "dilation_erosion":
            print(f"Perturbation Level: ±{level} mm (offset range)")
        else:
            print(f"Perturbation Level: σ={level} mm (noise std)")
        print("="*70)
        
        # Initialize data module with specific perturbation level
        # NOTE: seed is NOT passed to ensure different perturbations for each patient
        data_module = CoronaryArteryDataModuleSDMPerturbation(
            data_dir=data_dir,
            batch_size=1,
            patch_size=(96, 96, 96),
            num_workers=4,
            cache_rate=0.0,  # No caching for testing
            perturbation_mode=perturbation_mode,
            offset_range=level if perturbation_mode == "dilation_erosion" else 2.0,
            noise_std=level if perturbation_mode == "boundary_noise" else 0.5,
            perturbation_only_test=True,
            seed=None  # Each patient gets different perturbation
        )
        
        data_module.prepare_data()
        data_module.setup(stage="test")
        
        # Evaluate
        summary, detailed_results = evaluate_with_perturbation(
            model,
            data_module,
            device,
            perturbation_mode,
            level,
            output_path,
            save_predictions
        )
        
        all_summaries.append(summary)
        
        # Print summary for this level
        print(f"\nSummary for {perturbation_mode} level {level}:")
        print(f"  Dice:      {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"  Hausdorff: {summary['mean_hausdorff']:.4f} ± {summary['std_hausdorff']:.4f}")
        print(f"  IoU:       {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
        print(f"  Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
        print(f"  Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
        print(f"  clDice:    {summary['mean_cldice']:.4f} ± {summary['std_cldice']:.4f}")
        print(f"  Betti-0:   {summary['mean_betti_0']:.4f} ± {summary['std_betti_0']:.4f}")
        print(f"  Betti-1:   {summary['mean_betti_1']:.4f} ± {summary['std_betti_1']:.4f}")
    
    # Save summary across all levels
    summary_file = save_summary_results(all_summaries, output_path, arch_name, perturbation_mode)
    
    # Print final summary table
    print("\n\n" + "="*70)
    print(f"FINAL SUMMARY - Robustness to {perturbation_mode}")
    print("="*70)
    print(f"Architecture: {arch_name}")
    print(f"Perturbation Mode: {perturbation_mode}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\n")
    print(f"{'Level':<12} {'Dice':<20} {'Hausdorff':<20} {'clDice':<20}")
    print("-" * 70)
    
    for summary in all_summaries:
        level_str = f"{summary['perturbation_level']}"
        dice_str = f"{summary['mean_dice']:.4f}±{summary['std_dice']:.4f}"
        hausdorff_str = f"{summary['mean_hausdorff']:.2f}±{summary['std_hausdorff']:.2f}"
        cldice_str = f"{summary['mean_cldice']:.4f}±{summary['std_cldice']:.4f}"
        print(f"{level_str:<12} {dice_str:<20} {hausdorff_str:<20} {cldice_str:<20}")
    
    # Calculate performance degradation
    if len(all_summaries) > 1:
        baseline_dice = all_summaries[0]["mean_dice"]
        baseline_cldice = all_summaries[0]["mean_cldice"]
        
        print("\n" + "="*70)
        print("Performance Degradation (relative to baseline)")
        print("="*70)
        print(f"{'Level':<12} {'Dice Drop':<20} {'% Degradation':<20}")
        print("-" * 70)
        
        for summary in all_summaries[1:]:
            level_str = f"{summary['perturbation_level']}"
            dice_drop = baseline_dice - summary['mean_dice']
            percent_degrade = (dice_drop / baseline_dice) * 100 if baseline_dice > 0 else 0
            dice_drop_str = f"{dice_drop:.4f}"
            percent_str = f"{percent_degrade:.2f}%"
            print(f"{level_str:<12} {dice_drop_str:<20} {percent_str:<20}")
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"Summary file: {summary_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
