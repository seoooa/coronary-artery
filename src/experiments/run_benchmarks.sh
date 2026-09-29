#!/bin/bash

# Benchmark script runner
# Run benchmarks for both baseline and proposed models

echo "======================================"
echo "Running Inference Benchmarks"
echo "======================================"

# Running Code Examples:
# bash src/experiments/run_benchmarks.sh SegResNet 0 10 3
# bash src/experiments/run_benchmarks.sh UNETR 0 10 3
# bash src/experiments/run_benchmarks.sh SwinUNETR 1 20 5

# Configuration
ARCH_NAME=${1:-SegResNet}
GPU_NUMBER=${2:-0}
NUM_RUNS=${3:-10}
WARMUP_RUNS=${4:-3}

echo "Architecture: $ARCH_NAME"
echo "GPU Number: $GPU_NUMBER"
echo "Number of Runs: $NUM_RUNS"
echo "Warmup Runs: $WARMUP_RUNS"
echo ""

# Baseline model benchmark
echo ""
echo "Benchmarking Baseline $ARCH_NAME..."
python src/experiments/benchmark_baseline.py \
    --arch_name $ARCH_NAME \
    --checkpoint_path result/${ARCH_NAME}/final_model.ckpt \
    --gpu_number $GPU_NUMBER \
    --num_runs $NUM_RUNS \
    --warmup_runs $WARMUP_RUNS

# Proposed model with distanceMap
echo ""
echo "Benchmarking Proposed $ARCH_NAME (distanceMap)..."
python src/experiments/benchmark_proposed.py \
    --arch_name $ARCH_NAME \
    --checkpoint_path result/proposed_${ARCH_NAME}_dstMap/final_model.ckpt \
    --guide distanceMap \
    --gpu_number $GPU_NUMBER \
    --num_runs $NUM_RUNS \
    --warmup_runs $WARMUP_RUNS

# Proposed model with segMap (optional - uncomment if needed)
# echo ""
# echo "Benchmarking Proposed $ARCH_NAME (segMap)..."
# python src/experiments/benchmark_proposed.py \
#     --arch_name $ARCH_NAME \
#     --checkpoint_path result/proposed_${ARCH_NAME}_segMap/final_model.ckpt \
#     --guide segMap \
#     --gpu_number $GPU_NUMBER \
#     --num_runs $NUM_RUNS \
#     --warmup_runs $WARMUP_RUNS

echo ""
echo "======================================"
echo "Benchmarking Complete!"
echo "Results saved to: result/benchmarks/"
echo "======================================"
