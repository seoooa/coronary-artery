"""
Benchmark script for proposed model (proposed_train.py)
Measures:
- Inference time per patch
- Inference time per volume
- Peak GPU memory usage
"""

import autorootcwd
import torch
import time
import numpy as np
import click
from pathlib import Path
import csv
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureType,
    AsDiscrete,
)
from monai.data import decollate_batch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.proposed_dataloader import CoronaryArteryDataModule
from script.proposed_train import CoronaryArterySegmentModel

# FLOPs 계산을 위한 라이브러리
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install fvcore")


def measure_gpu_memory(device=None):
    """Measure current and peak GPU memory usage in GB"""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device = device.index if device.index is not None else 0
        
        current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        return current_memory, peak_memory
    return 0, 0


def calculate_flops(model, image, seg):
    """
    Calculate FLOPs for the model using fvcore
    
    Args:
        model: The model to analyze
        image: Input image tensor
        seg: Segmentation/distance map guide tensor
        
    Returns:
        Total FLOPs (in billions, GFLOPs) and detailed table
    """
    if not FVCORE_AVAILABLE:
        return None, "fvcore not available"
    
    try:
        # FLOPs 분석
        flops = FlopCountAnalysis(model, (image, seg))
        total_flops = flops.total()
        
        # GFLOPs로 변환 (10^9)
        gflops = total_flops / 1e9
        
        # 상세 테이블 생성
        flops_table = flop_count_table(flops)
        
        return gflops, flops_table
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None, str(e)


def benchmark_single_patch(model, image, seg, num_runs=10, warmup_runs=3):
    """
    Benchmark inference time for a single patch
    
    Args:
        model: The model to benchmark
        image: Single patch tensor
        seg: Segmentation/distance map guide
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Mean inference time in seconds
    """
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(image, seg)
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(image, seg)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def benchmark_full_volume(model, image, seg, patch_size=(96, 96, 96), num_runs=10, warmup_runs=3):
    """
    Benchmark inference time for a full volume using sliding window
    
    Args:
        model: The model to benchmark
        image: Full volume tensor
        seg: Segmentation/distance map guide
        patch_size: Patch size for sliding window
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Mean inference time in seconds
    """
    model.eval()
    sw_batch_size = 4
    
    # Concatenate image and seg
    inputs = torch.cat((image, seg), dim=1)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = sliding_window_inference(
                inputs,
                patch_size,
                sw_batch_size,
                lambda x: model(x[:, :1, ...], x[:, 1:, ...])
            )
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = sliding_window_inference(
                inputs,
                patch_size,
                sw_batch_size,
                lambda x: model(x[:, :1, ...], x[:, 1:, ...])
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(["SegResNet", "UNETR", "SwinUNETR", "nnFormer", "CSNet3D", "DSCNet", "AttentionUnet", "VNet"]),
    default="SegResNet",
    help="Choose the architecture name for the model.",
)
@click.option(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to a checkpoint file to load for benchmarking.",
)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="distanceMap",
    help="Choose the guide for training.",
)
@click.option(
    "--gpu_number",
    type=str,
    default="0",
    help="GPU number to use"
)
@click.option(
    "--num_runs",
    type=int,
    default=10,
    help="Number of benchmark runs"
)
@click.option(
    "--warmup_runs",
    type=int,
    default=3,
    help="Number of warmup runs"
)
@click.option(
    "--output_dir",
    type=str,
    default="result/experiments/benchmarks",
    help="Output directory for benchmark results"
)
def main(arch_name, checkpoint_path, guide, gpu_number, num_runs, warmup_runs, output_dir):
    # Set GPU
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = CoronaryArterySegmentModel.load_from_checkpoint(
        checkpoint_path,
        arch_name=arch_name,
        loss_fn="DiceFocalLoss",  # Doesn't matter for inference
        batch_size=1
    )
    model = model.to(device)
    model.eval()
    
    # Initialize data module to get test data
    data_module = CoronaryArteryDataModule(
        data_dir="data/imageCAS",
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=0,  # Use 0 for benchmarking
        cache_rate=0.0,
        use_distance_map=(guide == "distanceMap")
    )
    data_module.prepare_data()
    data_module.setup(stage="test")
    
    # Get a test sample
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))
    full_image = batch["image"].to(device)
    full_seg = batch["seg"].to(device)
    
    print(f"Full image shape: {full_image.shape}")
    print(f"Full seg shape: {full_seg.shape}")
    print(f"Guide type: {guide}")
    
    # Benchmark single patch
    print("\n" + "="*50)
    print("Benchmarking single patch inference...")
    print("="*50)
    
    # Create a single patch
    patch_size = (96, 96, 96)
    single_patch_image = full_image[:, :, :patch_size[0], :patch_size[1], :patch_size[2]]
    single_patch_seg = full_seg[:, :, :patch_size[0], :patch_size[1], :patch_size[2]]
    
    # Calculate FLOPs for single patch
    print("\nCalculating FLOPs for single patch...")
    patch_gflops, patch_flops_table = calculate_flops(
        model._model, single_patch_image, single_patch_seg
    )
    if patch_gflops is not None:
        print(f"Single patch FLOPs: {patch_gflops:.4f} GFLOPs")
        print("\nDetailed FLOPs breakdown:")
        print(patch_flops_table)
    
    # Reset GPU memory stats before benchmark
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    patch_time_mean, patch_time_std = benchmark_single_patch(
        model._model, single_patch_image, single_patch_seg, num_runs, warmup_runs
    )
    
    # Measure peak memory immediately after benchmark (before cleanup)
    _, patch_peak_memory = measure_gpu_memory(device)
    
    print(f"\nSingle patch inference time: {patch_time_mean:.4f} ± {patch_time_std:.4f} seconds")
    print(f"Peak GPU memory (single patch): {patch_peak_memory:.4f} GB")
    
    # Benchmark full volume
    print("\n" + "="*50)
    print("Benchmarking full volume inference...")
    print("="*50)
    
    # Reset GPU memory stats before benchmark
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    volume_time_mean, volume_time_std = benchmark_full_volume(
        model._model, full_image, full_seg, patch_size, num_runs, warmup_runs
    )
    
    # Measure peak memory immediately after benchmark (before cleanup)
    _, volume_peak_memory = measure_gpu_memory(device)
    
    print(f"Full volume inference time: {volume_time_mean:.4f} ± {volume_time_std:.4f} seconds")
    print(f"Peak GPU memory (full volume): {volume_peak_memory:.4f} GB")
    
    # Save results to CSV
    guide_suffix = "dstMap" if guide == "distanceMap" else "segMap"
    result_file = output_path / f"proposed_{arch_name}_{guide_suffix}_benchmark.csv"
    
    results = {
        "Architecture": arch_name,
        "Model Type": f"Proposed ({guide})",
        "Checkpoint": checkpoint_path,
        "Patch Inference Time (s)": f"{patch_time_mean:.4f} ± {patch_time_std:.4f}",
        "Volume Inference Time (s)": f"{volume_time_mean:.4f} ± {volume_time_std:.4f}",
        "Patch Peak GPU Memory (GB)": f"{patch_peak_memory:.4f}",
        "Volume Peak GPU Memory (GB)": f"{volume_peak_memory:.4f}",
        "Patch FLOPs (GFLOPs)": f"{patch_gflops:.4f}" if patch_gflops is not None else "N/A",
        "Num Runs": num_runs,
        "Warmup Runs": warmup_runs,
        "Volume Shape": str(full_image.shape),
        "Patch Size": str(patch_size),
        "Guide Type": guide,
    }
    
    with open(result_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    # Save detailed FLOPs table if available
    if patch_gflops is not None and patch_flops_table:
        flops_detail_file = output_path / f"proposed_{arch_name}_{guide_suffix}_flops_detail.txt"
        with open(flops_detail_file, "w") as f:
            f.write("="*80 + "\n")
            f.write(f"FLOPs Analysis for {arch_name} (Proposed {guide})\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total FLOPs: {patch_gflops:.4f} GFLOPs\n\n")
            f.write("Detailed breakdown by layer:\n")
            f.write(patch_flops_table)
        print(f"Detailed FLOPs analysis saved to: {flops_detail_file}")
    
    print("\n" + "="*50)
    print("Benchmark Summary")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
