"""
SPADE Hidden Layer Ablation Study
SegResNet에서 SPADE의 hidden layer 크기에 따른 성능 변화를 측정합니다.
"""

import autorootcwd
import os
import sys
import csv
import click
import torch
import subprocess
from pathlib import Path
import time

# SPADE hidden layer 크기별 설정
SPADE_CONFIGS = [
    {"hidden": 16, "module": "spade16"},
    {"hidden": 32, "module": "spade32"},
    {"hidden": 64, "module": "spade64"},
    {"hidden": 128, "module": "spade"},
]


def modify_segresnet_import(spade_module_name):
    """
    segresnet.py의 SPADE import 문을 동적으로 변경
    
    Args:
        spade_module_name: 사용할 SPADE 모듈 이름 (예: 'spade16', 'spade32', 'spade64', 'spade')
    """
    segresnet_path = Path("src/models/proposed/segresnet.py")
    
    # Read the file
    with open(segresnet_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace the import statement
    old_import = "from src.models.proposed.spade import SPADE"
    new_import = f"from src.models.proposed.{spade_module_name} import SPADE"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
    else:
        # Try to replace any existing modified import
        for config in SPADE_CONFIGS:
            temp_import = f"from src.models.proposed.{config['module']} import SPADE"
            if temp_import in content:
                content = content.replace(temp_import, new_import)
                break
    
    # Write back
    with open(segresnet_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✓ Modified segresnet.py to use {spade_module_name}")


def train_model(hidden_size, spade_module, max_epochs, gpu_number, guide="distanceMap"):
    """
    주어진 SPADE hidden 크기로 모델 학습
    
    Args:
        hidden_size: SPADE hidden layer 크기
        spade_module: SPADE 모듈 이름
        max_epochs: 최대 에포크 수
        gpu_number: 사용할 GPU 번호
        guide: 가이드 타입 (distanceMap or segMap)
    
    Returns:
        checkpoint_path: 저장된 모델 경로
    """
    print("\n" + "="*70)
    print(f"Training SegResNet with SPADE hidden={hidden_size}")
    print("="*70)
    
    # Modify segresnet.py import
    modify_segresnet_import(spade_module)
    
    # Training command
    cmd = [
        "python", "script/proposed_train.py",
        "--arch_name", "SegResNet",
        "--loss_fn", "DiceFocalLoss",
        "--max_epochs", str(max_epochs),
        "--check_val_every_n_epoch", "10",
        "--gpu_number", str(gpu_number),
        "--guide", guide,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"✗ Training failed for hidden={hidden_size}")
        return None
    
    print(f"✓ Training completed in {elapsed_time:.2f} seconds")
    
    # Find checkpoint path
    guide_suffix = "_dstMap" if guide == "distanceMap" else "_segMap"
    checkpoint_dir = Path(f"result/proposed_SegResNet{guide_suffix}_DiceFocalLoss")
    checkpoint_path = checkpoint_dir / "final_model.ckpt"
    
    if checkpoint_path.exists():
        # Rename checkpoint to include hidden size
        new_checkpoint_path = checkpoint_dir / f"final_model_spade{hidden_size}.ckpt"
        checkpoint_path.rename(new_checkpoint_path)
        print(f"✓ Checkpoint saved: {new_checkpoint_path}")
        return str(new_checkpoint_path), elapsed_time
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None, elapsed_time


def evaluate_model(checkpoint_path, hidden_size, spade_module, gpu_number, guide="distanceMap"):
    """
    저장된 모델로 테스트 데이터셋 평가
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        hidden_size: SPADE hidden layer 크기
        spade_module: SPADE 모듈 이름
        gpu_number: 사용할 GPU 번호
        guide: 가이드 타입
    
    Returns:
        metrics: 평가 지표 딕셔너리
    """
    print("\n" + "="*70)
    print(f"Evaluating SegResNet with SPADE hidden={hidden_size}")
    print("="*70)
    
    # Modify segresnet.py import
    modify_segresnet_import(spade_module)
    
    # Evaluation command
    cmd = [
        "python", "script/proposed_train.py",
        "--arch_name", "SegResNet",
        "--loss_fn", "DiceFocalLoss",
        "--gpu_number", str(gpu_number),
        "--checkpoint_path", checkpoint_path,
        "--guide", guide,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"✗ Evaluation failed for hidden={hidden_size}")
        return None
    
    print(f"✓ Evaluation completed")
    
    # Read test results
    guide_suffix = "_dstMap" if guide == "distanceMap" else "_segMap"
    result_dir = Path(f"result/experiments/spade_ablation/proposed_SegResNet{guide_suffix}_DiceFocalLoss/test")
    result_file = result_dir / "test_result.csv"
    
    if not result_file.exists():
        print(f"✗ Result file not found: {result_file}")
        return None
    
    # Parse results
    with open(result_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Get the summary row (last row with AVG ± STD)
        summary_row = rows[-1]
        
        metrics = {
            "dice": summary_row["dice_score"],
            "hausdorff": summary_row["hausdorff_score"],
            "iou": summary_row["iou_score"],
            "precision": summary_row["precision_score"],
            "recall": summary_row["recall_score"],
            "cldice": summary_row["cldice_score"],
            "betti_0": summary_row["betti_0_score"],
            "betti_1": summary_row["betti_1_score"],
        }
    
    print(f"✓ Metrics extracted: Dice={metrics['dice']}")
    return metrics


@click.command()
@click.option(
    "--max_epochs",
    type=int,
    default=200,
    help="Maximum number of training epochs"
)
@click.option(
    "--gpu_number",
    type=str,
    default="0",
    help="GPU number to use"
)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="distanceMap",
    help="Guide type for training"
)
@click.option(
    "--skip_training",
    is_flag=True,
    help="Skip training and only evaluate existing checkpoints"
)
@click.option(
    "--output_dir",
    type=str,
    default="result/experiments/spade_ablation",
    help="Output directory for results"
)
def main(max_epochs, gpu_number, guide, skip_training, output_dir):
    """
    SPADE Hidden Layer Ablation Study
    
    다양한 hidden layer 크기 (16, 32, 64, 128)로 SegResNet을 학습하고 성능을 비교합니다.
    """
    print("="*70)
    print("SPADE HIDDEN LAYER ABLATION STUDY")
    print("="*70)
    print(f"GPU: {gpu_number}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Guide Type: {guide}")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    all_results = []
    
    for config in SPADE_CONFIGS:
        hidden_size = config["hidden"]
        spade_module = config["module"]
        
        result = {
            "hidden_size": hidden_size,
            "spade_module": spade_module,
        }
        
        # Checkpoint path
        guide_suffix = "_dstMap" if guide == "distanceMap" else "_segMap"
        checkpoint_dir = Path(f"result/experiments/spade_ablation/proposed_SegResNet{guide_suffix}_DiceFocalLoss")
        checkpoint_path = checkpoint_dir / f"final_model_spade{hidden_size}.ckpt"
        
        # Train model
        if not skip_training:
            ckpt_path, train_time = train_model(
                hidden_size, spade_module, max_epochs, gpu_number, guide
            )
            result["training_time"] = f"{train_time:.2f}"
            
            if ckpt_path is None:
                print(f"Skipping evaluation for hidden={hidden_size} due to training failure")
                continue
            
            checkpoint_path = Path(ckpt_path)
        else:
            if not checkpoint_path.exists():
                print(f"✗ Checkpoint not found: {checkpoint_path}")
                print(f"Skipping hidden={hidden_size}")
                continue
            result["training_time"] = "N/A (skipped)"
        
        # Evaluate model
        metrics = evaluate_model(
            str(checkpoint_path), hidden_size, spade_module, gpu_number, guide
        )
        
        if metrics is None:
            print(f"Skipping hidden={hidden_size} due to evaluation failure")
            continue
        
        # Store results
        result.update(metrics)
        all_results.append(result)
    
    # Save results to CSV
    if all_results:
        csv_file = Path(output_dir) / "spade_ablation_results.csv"
        
        with open(csv_file, "w", newline="") as f:
            fieldnames = [
                "hidden_size",
                "spade_module",
                "training_time",
                "dice",
                "hausdorff",
                "iou",
                "precision",
                "recall",
                "cldice",
                "betti_0",
                "betti_1",
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in all_results:
                writer.writerow(result)
        
        print("\n" + "="*70)
        print("ABLATION STUDY COMPLETED")
        print("="*70)
        print(f"Results saved to: {csv_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"{'Hidden Size':<15} {'Dice Score':<20} {'Training Time':<20}")
        print("-" * 70)
        for result in all_results:
            print(f"{result['hidden_size']:<15} {result['dice']:<20} {result['training_time']:<20}")
        print("="*70)
    else:
        print("\n✗ No results to save")


if __name__ == "__main__":
    main()
