#!/bin/bash

# SPADE Hidden Layer Ablation Study 실행 스크립트
# SegResNet에서 SPADE의 hidden layer 크기에 따른 성능을 비교합니다.

# 기본 설정
GPU_NUMBER=${1:-0}
MAX_EPOCHS=${2:-200}
GUIDE=${3:-distanceMap}

echo "======================================================================"
echo "SPADE Hidden Layer Ablation Study"
echo "======================================================================"
echo "GPU: $GPU_NUMBER"
echo "Max Epochs: $MAX_EPOCHS"
echo "Guide Type: $GUIDE"
echo "======================================================================"

# 전체 학습 + 평가 실행
python src/experiments/spade_ablation_study.py \
    --max_epochs $MAX_EPOCHS \
    --gpu_number $GPU_NUMBER \
    --guide $GUIDE \
    --output_dir result/experiments/spade_ablation

echo "======================================================================"
echo "Ablation study completed!"
echo "Results saved to: result/experiments/spade_ablation/"
echo "======================================================================"
