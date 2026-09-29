#!/bin/bash

# TotalSegmentator 전처리 벤치마크 실행 스크립트

echo "======================================"
echo "TotalSegmentator Heart Preprocessing Benchmark"
echo "======================================"

# Running Code Examples:
# bash src/experiments/run_preprocessing_time.sh 5 4
# bash src/experiments/run_preprocessing_time.sh 10 0

# Configuration
NUM_PATIENTS=${1:-5}
GPU_NUMBER=${2:-4}

echo "Number of Patients: $NUM_PATIENTS"
echo "GPU Number: $GPU_NUMBER"
echo "ROI: Heart (Coronary Arteries + Heart Chambers)"
echo ""

# Run benchmark for Heart ROI only
python src/experiments/preprocessing_time.py \
    --input_dir data/imageCAS/test \
    --output_root result/experiments/preprocessing \
    --num_patients $NUM_PATIENTS \
    --gpu_number $GPU_NUMBER

echo ""
echo "======================================"
echo "Heart Preprocessing Benchmark Complete!"
echo "======================================"
echo "Results saved to: result/experiments/preprocessing/"
echo ""
