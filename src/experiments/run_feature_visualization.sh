#!/bin/bash

# SPADE Feature Visualization 실행 스크립트
# Decoder feature map 시각화 및 t-SNE 분석

# 기본 설정
BASELINE_CKPT=${1:-"result/SegResNet_DiceFocalLoss/final_model.ckpt"}
PROPOSED_CKPT=${2:-"result/proposed_SegResNet_dstMap_DiceFocalLoss/final_model.ckpt"}
DATA_DIR=${3:-"data/imageCAS/test"}
NUM_PATIENTS=${4:-5}
GPU_NUMBER=${5:-0}
OUTPUT_DIR=${6:-"result/experiments/feature_visualization"}

echo "======================================================================"
echo "SPADE Feature Visualization"
echo "======================================================================"
echo "Baseline Checkpoint: $BASELINE_CKPT"
echo "Proposed Checkpoint: $PROPOSED_CKPT"
echo "Data Directory: $DATA_DIR"
echo "Number of Patients: $NUM_PATIENTS"
echo "GPU: $GPU_NUMBER"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================"

# 체크포인트 파일 존재 확인
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "Error: Baseline checkpoint not found: $BASELINE_CKPT"
    exit 1
fi

if [ ! -f "$PROPOSED_CKPT" ]; then
    echo "Error: Proposed checkpoint not found: $PROPOSED_CKPT"
    exit 1
fi

# 데이터 디렉토리 존재 확인
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Feature visualization 실행
python src/experiments/feature_visualization.py \
    --baseline_ckpt "$BASELINE_CKPT" \
    --proposed_ckpt "$PROPOSED_CKPT" \
    --data_dir "$DATA_DIR" \
    --num_patients $NUM_PATIENTS \
    --output_dir "$OUTPUT_DIR" \
    --gpu_number $GPU_NUMBER \
    --num_samples 1000 \
    --perplexity 30

echo ""
echo "======================================================================"
echo "Feature visualization completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================================================"
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── summary.json                      # Silhouette score 요약"
echo "    ├── patient_XXX/"
echo "    │   ├── patient_XXX_layer_*.png       # Feature map 시각화"
echo "    │   └── patient_XXX_tsne.png          # t-SNE 클러스터링"
echo "    └── ..."
echo "======================================================================"
