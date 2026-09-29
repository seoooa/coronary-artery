# SPADE Hidden Layer Ablation Study

SegResNet에서 **SPADE의 hidden layer 크기**에 따른 성능 변화를 측정하는 실험입니다.

## 📋 실험 개요

### 목적
SPADE 정규화 레이어의 hidden layer 크기가 관상동맥 분할 성능에 미치는 영향을 분석합니다.

### 실험 조건
- **모델**: SegResNet (SPADE 정규화 포함)
- **Hidden Layer 크기**: 16, 32, 64, 128
- **손실 함수**: DiceFocalLoss
- **Guide 타입**: distanceMap (기본) 또는 segMap
- **데이터셋**: imageCAS

### 실험 구조
```
src/models/proposed/
├── spade16.py      # nhidden = 16
├── spade32.py      # nhidden = 32
├── spade64.py      # nhidden = 64
└── spade.py        # nhidden = 128 (기본)
```

## 🚀 사용 방법

### 1. 기본 실행 (전체 학습 + 평가)

```bash
bash src/experiments/run_spade_ablation.sh [GPU_NUMBER] [MAX_EPOCHS] [GUIDE_TYPE]
```

**예시:**
```bash
# GPU 0번, 200 에포크, distanceMap 가이드
bash src/experiments/run_spade_ablation.sh 0 200 distanceMap

# GPU 4번, 100 에포크, segMap 가이드
bash src/experiments/run_spade_ablation.sh 4 100 segMap
```

### 2. 직접 Python 스크립트 실행

```bash
python src/experiments/spade_ablation_study.py \
    --max_epochs 200 \
    --gpu_number 0 \
    --guide distanceMap \
    --output_dir result/experiments/spade_ablation
```

### 3. 기존 체크포인트로 평가만 수행

이미 학습된 모델이 있는 경우:

```bash
python src/experiments/spade_ablation_study.py \
    --skip_training \
    --gpu_number 0 \
    --guide distanceMap \
    --output_dir result/experiments/spade_ablation
```

## 📊 결과 파일

### 1. 통합 결과 CSV
**경로**: `result/experiments/spade_ablation/spade_ablation_results.csv`

**포함 내역**:
```csv
hidden_size,spade_module,training_time,dice,hausdorff,iou,precision,recall,cldice,betti_0,betti_1
16,spade16,3600.50,0.7234 ± 0.0123,...
32,spade32,3612.20,0.7456 ± 0.0098,...
64,spade64,3625.80,0.7523 ± 0.0105,...
128,spade,3680.40,0.7601 ± 0.0092,...
```

### 2. 개별 모델 결과
각 hidden 크기별로 다음 파일들이 생성됩니다:

```
result/proposed_SegResNet_dstMap_DiceFocalLoss/
├── final_model_spade16.ckpt          # 모델 체크포인트
├── final_model_spade32.ckpt
├── final_model_spade64.ckpt
├── final_model_spade128.ckpt
└── test/
    ├── test_result.csv               # 상세 테스트 결과
    ├── Subj_XXX_inputs.nii.gz       # 입력 이미지
    ├── Subj_XXX_outputs.nii.gz      # 예측 결과
    └── Subj_XXX_labels.nii.gz       # 정답 레이블
```

## 📈 평가 지표

각 모델에 대해 다음 지표들이 측정됩니다:

| 지표 | 설명 |
|------|------|
| **Dice Score** | 세그멘테이션 정확도 (0~1, 높을수록 좋음) |
| **Hausdorff Distance** | 경계 일치 정확도 (낮을수록 좋음) |
| **IoU** | Intersection over Union (0~1, 높을수록 좋음) |
| **Precision** | 정밀도 (0~1, 높을수록 좋음) |
| **Recall** | 재현율 (0~1, 높을수록 좋음) |
| **CLDice** | 중심선 Dice Score (0~1, 높을수록 좋음) |
| **Betti-0 Error** | 연결 성분 수 오차 (낮을수록 좋음) |
| **Betti-1 Error** | 루프 개수 오차 (낮을수록 좋음) |

## 🔬 실험 프로세스

1. **SPADE 모듈 생성**: 각 hidden 크기별 SPADE 모듈 준비
2. **SegResNet 수정**: `segresnet.py`의 import 문을 동적으로 변경
3. **모델 학습**: 각 설정으로 SegResNet 학습 (200 에포크)
4. **모델 평가**: 테스트 데이터셋으로 성능 측정
5. **결과 저장**: CSV 파일로 통합 결과 저장

## ⚙️ 커맨드 라인 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--max_epochs` | int | 200 | 최대 학습 에포크 수 |
| `--gpu_number` | str | "0" | 사용할 GPU 번호 (예: "0" 또는 "0,1,2,3") |
| `--guide` | str | "distanceMap" | 가이드 타입 (distanceMap / segMap) |
| `--skip_training` | flag | False | 학습 건너뛰고 평가만 수행 |
| `--output_dir` | str | result/experiments/spade_ablation | 결과 저장 디렉토리 |

## 💡 팁

### 1. 빠른 테스트
학습 시간을 줄이고 싶다면:
```bash
bash src/experiments/run_spade_ablation.sh 0 50 distanceMap
```

### 2. 특정 hidden 크기만 테스트
`spade_ablation_study.py`의 `SPADE_CONFIGS` 리스트를 수정:
```python
SPADE_CONFIGS = [
    {"hidden": 32, "module": "spade32"},
    {"hidden": 64, "module": "spade64"},
]
```

### 3. 멀티 GPU 학습
```bash
python src/experiments/spade_ablation_study.py \
    --gpu_number "0,1,2,3" \
    --max_epochs 200
```

## 📝 주의사항

1. **Import 문 자동 수정**: 스크립트가 자동으로 `segresnet.py`의 SPADE import 문을 수정합니다.
2. **원본 복구**: 실험 종료 후 원하는 SPADE 모듈로 되돌리려면 수동으로 수정하세요.
3. **디스크 공간**: 각 모델마다 체크포인트와 테스트 결과가 저장되므로 충분한 공간 필요
4. **학습 시간**: 4개 hidden 크기 × 200 에포크 = 약 12~24시간 소요 (GPU 성능에 따라 다름)

## 🔍 결과 분석 예시

실험 완료 후 결과를 분석하려면:

```python
import pandas as pd

# 결과 로드
df = pd.read_csv("result/experiments/spade_ablation/spade_ablation_results.csv")

# Dice Score 비교
print(df[["hidden_size", "dice"]].sort_values("dice", ascending=False))

# 학습 시간 vs 성능
import matplotlib.pyplot as plt
plt.scatter(df["training_time"], df["dice"])
plt.xlabel("Training Time (s)")
plt.ylabel("Dice Score")
plt.title("SPADE Hidden Size: Training Time vs Performance")
plt.show()
```

## 📚 참고 문헌

- **SPADE**: Semantic Image Synthesis with Spatially-Adaptive Normalization
- **SegResNet**: 3D MRI brain tumor segmentation using autoencoder regularization

## 🐛 문제 해결

### Q: "Checkpoint not found" 오류
**A**: `--skip_training` 플래그 없이 먼저 학습을 완료하세요.

### Q: GPU 메모리 부족
**A**: `proposed_train.py`의 배치 크기나 patch 크기를 줄이세요.

### Q: Import 오류
**A**: 실험 전 모든 SPADE 모듈이 존재하는지 확인하세요:
```bash
ls src/models/proposed/spade*.py
```

---

**작성일**: 2026-01-26  
**작성자**: SPADE Ablation Study Script
