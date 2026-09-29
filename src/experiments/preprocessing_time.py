"""
TotalSegmentator Heart 전처리 시간 벤치마크 스크립트
Heart ROI (관상동맥 + 심장) 분할 시간을 측정합니다.
"""

import autorootcwd
import torch
import time
import numpy as np
import click
from pathlib import Path
import csv
import os
import sys
import shutil
from tqdm import tqdm

# TotalSegmentator 직접 import
from totalsegmentator.python_api import totalsegmentator

# roi_segmentation.py의 병합 함수만 import
sys.path.append(str(Path(__file__).parent.parent))
from src.data.roi_segmentation import combine_segmentations


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


def benchmark_single_patient(patient_input_path, output_dir, device):
    """
    단일 환자에 대한 Heart 전처리 시간 측정
    
    Args:
        patient_input_path: 입력 이미지 경로 (img.nii.gz)
        output_dir: 출력 디렉토리
        device: PyTorch device 객체
        
    Returns:
        Dictionary containing timing results for Heart ROI
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract device index for CUDA API calls
    # After CUDA_VISIBLE_DEVICES is set, device index is always 0
    device_idx = 0
    
    results = {
        "heart": {"time": 0, "success": False, "peak_memory": 0},
        "total": {"time": 0, "success": False, "peak_memory": 0}
    }
    
    total_start_time = time.time()
    
    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_idx)
        torch.cuda.empty_cache()
    
    # Heart segmentation
    print("\n" + "="*60)
    print("Benchmarking Heart Segmentation...")
    print("="*60)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_idx)
        torch.cuda.empty_cache()
    
    # Create temporary folder for TotalSegmentator output
    temp_folder = os.path.join(output_dir, "temp_heart")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Define label map for heart-related areas
    label_map = {
        "coronary_arteries.nii.gz": 1,
        "aorta.nii.gz": 2,
        "heart_myocardium.nii.gz": 3,
        "heart_ventricle_left.nii.gz": 4,
        "heart_ventricle_right.nii.gz": 5,
        "heart_atrium_left.nii.gz": 6,
        "heart_atrium_right.nii.gz": 7,
    }
    
    start_time = time.time()
    success = False
    
    try:
        # Run TotalSegmentator for coronary arteries
        print("Running TotalSegmentator: coronary_arteries task...")
        totalsegmentator(
            patient_input_path,
            temp_folder,
            task='coronary_arteries',
            device='gpu',
        )
        
        # Run TotalSegmentator for heart chambers
        print("Running TotalSegmentator: heartchambers_highres task...")
        totalsegmentator(
            patient_input_path,
            temp_folder,
            task='heartchambers_highres',
            device='gpu',
        )
        
        # Check files in temporary folder
        print("\nFiles in temporary folder:")
        temp_files = os.listdir(temp_folder)
        for file in temp_files:
            print(f"  - {file}")
        
        # Combine segmentations
        combine_path = os.path.join(output_dir, "heart_combined.nii.gz")
        combine_segmentations(temp_folder, combine_path, label_map, is_flat=True)
        
        print("Heart segmentation completed!")
        success = True
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        _, peak_memory = measure_gpu_memory(device_idx)
        
        results["heart"]["time"] = elapsed_time
        results["heart"]["success"] = success
        results["heart"]["peak_memory"] = peak_memory
        
        print(f"Heart segmentation time: {elapsed_time:.2f} seconds")
        print(f"Heart peak GPU memory: {peak_memory:.4f} GB")
        
    except Exception as e:
        print(f"Heart segmentation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        results["heart"]["time"] = time.time() - start_time
        results["heart"]["success"] = False
        
    finally:
        # Clean up temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    results["total"]["time"] = total_elapsed_time
    results["total"]["success"] = results["heart"]["success"]
    results["total"]["peak_memory"] = results["heart"]["peak_memory"]
    
    print("\n" + "="*60)
    print(f"Total preprocessing time: {total_elapsed_time:.2f} seconds")
    print(f"Peak GPU memory: {results['heart']['peak_memory']:.4f} GB")
    print("="*60)
    
    return results


def benchmark_multiple_patients(input_dir, output_root, num_patients, device):
    """
    여러 환자에 대한 평균 Heart 전처리 시간 측정
    
    Args:
        input_dir: 입력 데이터 디렉토리 (환자 폴더들이 있는 곳)
        output_root: 출력 루트 디렉토리
        num_patients: 측정할 환자 수
        device: PyTorch device 객체
        
    Returns:
        List of results for each patient
    """
    # Get list of patient directories
    patient_dirs = sorted(os.listdir(input_dir))[:num_patients]
    
    all_results = []
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_input_path = os.path.join(input_dir, patient_dir, "img.nii.gz")
        patient_output_dir = os.path.join(output_root, patient_dir)
        
        if not os.path.exists(patient_input_path):
            print(f"\nSkipping {patient_dir} - img.nii.gz not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing patient: {patient_dir}")
        print(f"{'='*60}")
        
        results = benchmark_single_patient(
            patient_input_path, 
            patient_output_dir,
            device
        )
        
        results["patient_id"] = patient_dir
        all_results.append(results)
        
        # Clean up output directory to save space (optional)
        # shutil.rmtree(patient_output_dir, ignore_errors=True)
    
    return all_results


@click.command()
@click.option(
    "--input_dir",
    type=str,
    default="data/imageCAS/test",
    help="Input directory containing patient folders"
)
@click.option(
    "--output_root",
    type=str,
    default="result/experiments/preprocessing",
    help="Output root directory for benchmark results"
)
@click.option(
    "--num_patients",
    type=int,
    default=5,
    help="Number of patients to benchmark (default: 5)"
)
@click.option(
    "--gpu_number",
    type=int,
    default=0,
    help="GPU number to use (default: 0)"
)
def main(input_dir, output_root, num_patients, gpu_number):
    """
    TotalSegmentator Heart 전처리 시간 벤치마크
    Heart ROI (관상동맥 + 심장) 분할 시간만 측정합니다.
    """
    # Set CUDA device (same as roi_segmentation.py)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    
    # After setting CUDA_VISIBLE_DEVICES, use cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize CUDA context (important for memory stats)
    if torch.cuda.is_available():
        _ = torch.zeros(1).to(device)
        torch.cuda.synchronize()
    
    print("="*70)
    print("TOTALSEGMENTATOR HEART PREPROCESSING BENCHMARK")
    print("="*70)
    print(f"Using device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_root}")
    print(f"Number of patients: {num_patients}")
    print(f"Selected ROI: Heart (Coronary Arteries + Heart Chambers)")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Run benchmark
    all_results = benchmark_multiple_patients(
        input_dir, 
        output_root,
        num_patients,
        device
    )
    
    if not all_results:
        print("\nNo patients were successfully processed.")
        return
    
    # Calculate statistics
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    # Calculate mean and std for Heart and Total
    roi_stats = {}
    for roi in ["heart", "total"]:
        times = [r[roi]["time"] for r in all_results if r[roi]["success"]]
        memories = [r[roi]["peak_memory"] for r in all_results if r[roi]["success"]]
        
        if times:
            roi_stats[roi] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "mean_memory": np.mean(memories),
                "max_memory": np.max(memories),
                "success_rate": sum([r[roi]["success"] for r in all_results]) / len(all_results) * 100
            }
    
    # Print statistics
    for roi, stats in roi_stats.items():
        print(f"\n{roi.upper()}:")
        print(f"  Time: {stats['mean_time']:.2f} ± {stats['std_time']:.2f} seconds")
        print(f"  Mean GPU Memory: {stats['mean_memory']:.4f} GB")
        print(f"  Max GPU Memory: {stats['max_memory']:.4f} GB")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
    
    # Save detailed results to CSV
    csv_file = os.path.join(output_root, "preprocessing_benchmark_results.csv")
    
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "patient_id",
            "heart_time_s",
            "heart_memory_gb",
            "heart_success",
            "total_time_s",
            "total_max_memory_gb"
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write individual patient results
        for result in all_results:
            row = {
                "patient_id": result["patient_id"],
                "heart_time_s": f"{result['heart']['time']:.2f}",
                "heart_memory_gb": f"{result['heart']['peak_memory']:.4f}",
                "heart_success": result["heart"]["success"],
                "total_time_s": f"{result['total']['time']:.2f}",
                "total_max_memory_gb": f"{result['total']['peak_memory']:.4f}"
            }
            writer.writerow(row)
        
        # Write summary statistics
        if "heart" in roi_stats and "total" in roi_stats:
            summary_row = {
                "patient_id": "MEAN ± STD",
                "heart_time_s": f"{roi_stats['heart']['mean_time']:.2f} ± {roi_stats['heart']['std_time']:.2f}",
                "heart_memory_gb": f"{roi_stats['heart']['mean_memory']:.4f}",
                "heart_success": f"{roi_stats['heart']['success_rate']:.1f}%",
                "total_time_s": f"{roi_stats['total']['mean_time']:.2f} ± {roi_stats['total']['std_time']:.2f}",
                "total_max_memory_gb": f"{roi_stats['total']['max_memory']:.4f}"
            }
            writer.writerow(summary_row)
    
    print(f"\n{'='*70}")
    print(f"Detailed results saved to: {csv_file}")
    print("="*70)


if __name__ == "__main__":
    main()
