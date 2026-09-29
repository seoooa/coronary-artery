# SPADE Feature Visualization

Decoder의 각 layer에서 **SPADE 적용 전후의 feature map**을 시각화하고, **해부학적 위치에 따른 feature clustering**을 t-SNE로 분석하는 실험입니다.

## 📋 실험 개요

### 목적
1. **Feature Map 시각화**: Decoder의 각 layer에서 SPADE가 feature에 어떤 영향을 주는지 시각화
2. **t-SNE 분석**: 해부학적 위치에 따라 feature가 더 잘 군집화되는지 정량적으로 평가
3. **Silhouette Score**: 클러스터링 품질을 수치로 측정

### 비교 대상
- **Baseline Model**: SPADE 없는 SegResNet (`src/models/model/segresnet.py`)
- **Proposed Model**: SPADE 있는 SegResNet (`src/models/proposed/segresnet.py`)

## 🚀 사용 방법

### 1. 기본 실행

```bash
bash src/experiments/run_feature_visualization.sh \
    [BASELINE_CKPT] \
    [PROPOSED_CKPT] \
    [DATA_DIR] \
    [NUM_PATIENTS] \
    [GPU_NUMBER] \
    [OUTPUT_DIR]
```

**예시:**
```bash
# 기본 경로 사용
bash src/experiments/run_feature_visualization.sh

# 커스텀 경로 지정
bash src/experiments/run_feature_visualization.sh \
    result/SegResNet/final_model.ckpt \
    result/proposed_SegResNet_dstMap/final_model.ckpt \
    data/imageCAS/test \
    10 \
    5 \
    result/experiments/feature_visualization
```

### 2. Python 직접 실행

```bash
python src/experiments/feature_visualization.py \
    --baseline_ckpt result/SegResNet_DiceFocalLoss/final_model.ckpt \
    --proposed_ckpt result/proposed_SegResNet_dstMap_DiceFocalLoss/final_model.ckpt \
    --data_dir data/imageCAS/test \
    --num_patients 5 \
    --output_dir result/experiments/feature_visualization \
    --gpu_number 0 \
    --num_samples 1000 \
    --perplexity 30
```

### 3. 특정 환자 ID 지정 🆕

```bash
# 특정 환자 ID들만 처리 (예: 880, 980)
python src/experiments/feature_visualization.py \
    --baseline_ckpt result/SegResNet/final_model.ckpt \
    --proposed_ckpt result/proposed_SegResNet_dstMap/final_model.ckpt \
    --data_dir data/imageCAS/test \
    --patient_ids "755,758,802,841,845,863,925,930,980,990" \
    --output_dir result/experiments/feature_visualization \
    --gpu_number 6

# 여러 환자 추가 가능
python src/experiments/feature_visualization.py \
    --baseline_ckpt result/SegResNet_DiceFocalLoss/final_model.ckpt \
    --proposed_ckpt result/proposed_SegResNet_dstMap_DiceFocalLoss/final_model.ckpt \
    --data_dir data/imageCAS/test \
    --patient_ids "751,880,980,1000,1050" \
    --output_dir result/experiments/feature_visualization \
    --gpu_number 0
```

## 📊 출력 파일

### 디렉토리 구조
```
result/experiments/feature_visualization/
├── summary.json                              # 전체 요약 (2D/3D Silhouette scores)
├── patient_751/
│   ├── patient_751_layer_up_layers_0.png           # Decoder layer 0 feature map (3x3 layout)
│   ├── patient_751_layer_up_layers_1.png           # Decoder layer 1 feature map (3x3 layout)
│   ├── patient_751_layer_up_layers_2.png           # Decoder layer 2 feature map (3x3 layout)
│   ├── patient_751_tsne_2d_heart_all.png            # 2D t-SNE (All 7 heart labels)
│   ├── patient_751_tsne_3d_heart_all.png            # 3D t-SNE (All 7 heart labels)
│   ├── patient_751_tsne_2d_heart_coronary_arteries.png  # 2D t-SNE (Coronary vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_coronary_arteries.png  # 3D t-SNE (Coronary vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_aorta.png              # 2D t-SNE (Aorta vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_aorta.png              # 3D t-SNE (Aorta vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_myocardium.png         # 2D t-SNE (Myocardium vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_myocardium.png         # 3D t-SNE (Myocardium vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_left_ventricle.png     # 2D t-SNE (LV vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_left_ventricle.png     # 3D t-SNE (LV vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_right_ventricle.png    # 2D t-SNE (RV vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_right_ventricle.png    # 3D t-SNE (RV vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_left_atrium.png        # 2D t-SNE (LA vs Others) 🆕
│   ├── patient_751_tsne_3d_heart_left_atrium.png        # 3D t-SNE (LA vs Others) 🆕
│   ├── patient_751_tsne_2d_heart_right_atrium.png       # 2D t-SNE (RA vs Others) 🆕
│   └── patient_751_tsne_3d_heart_right_atrium.png       # 3D t-SNE (RA vs Others) 🆕
├── patient_752/
│   └── ...
└── ...

**Total: 3 feature maps + 16 t-SNE visualizations per patient**
```

### 1. Feature Map 시각화 (`patient_XXX_layer_*.png`)

각 이미지는 **3x3 layout**으로 구성:

**Row 1: Anatomical Context**
- **CT Image**: 원본 CT 이미지
- **Heart Segmentation**: 8개 해부학적 구조 (coronary arteries, aorta, myocardium, ventricles, atria)
- **Coronary Artery Label**: Ground truth segmentation (관상동맥)

**Row 2: Feature Maps**
- **Baseline Features**: SPADE 없는 모델의 feature activation
- **Proposed (SPADE) Features**: SPADE 있는 모델의 feature activation
- **Difference**: Feature 변화량 (Proposed - Baseline, 빨강/파랑으로 차이 표시)

**Row 3: Analysis & Overlays**
- **Absolute Difference**: 절대값 차이 (뜨거운 색일수록 큰 차이)
- **Feature Change on CT**: CT 이미지 위에 feature 변화 overlay
- **Feature Change with Coronary Context**: 관상동맥 위치(초록)와 feature 변화(빨강)를 함께 표시

### 2. t-SNE 시각화

#### A. Heart All (전체 7개 레이블 동시 시각화)

**파일**: `patient_XXX_tsne_2d_heart_all.png`, `patient_XXX_tsne_3d_heart_all.png`

- **Baseline Model**: 7개 해부학적 구조의 feature 분포
- **Proposed Model (SPADE)**: SPADE 적용 후 feature 분포

**색상** (7 classes, background 제외):
- Label 1: Coronary Arteries
- Label 2: Aorta
- Label 3: Myocardium
- Label 4: Left Ventricle
- Label 5: Right Ventricle
- Label 6: Left Atrium
- Label 7: Right Atrium

**의미**: SPADE가 다양한 해부학적 구조를 **동시에** 구분하는 능력

#### B. Heart Individual (각 레이블별 Binary 시각화) 🆕

**파일**: `patient_XXX_tsne_2d_heart_{label_name}.png`, `patient_XXX_tsne_3d_heart_{label_name}.png`

각 해부학적 구조별로 **타겟 vs Others** binary classification:
- `heart_coronary_arteries`: Coronary Arteries vs Others (Aorta, Myocardium, Ventricles, Atria)
- `heart_aorta`: Aorta vs Others
- `heart_myocardium`: Myocardium vs Others
- `heart_left_ventricle`: Left Ventricle vs Others
- `heart_right_ventricle`: Right Ventricle vs Others
- `heart_left_atrium`: Left Atrium vs Others
- `heart_right_atrium`: Right Atrium vs Others

**색상** (Binary, background 제외):
- Label 0 (파란색): Others (다른 모든 해부학적 구조)
- Label 1 (주황색): Target (해당 해부학적 구조)

**의미**: 
- 각 해부학적 구조의 **feature distinctiveness** (특징 고유성)
- Silhouette score가 높으면 → 해당 구조의 feature가 다른 구조와 명확히 구별됨
- SPADE가 특정 구조의 feature를 얼마나 잘 강조하는지 확인

### 3. 요약 파일 (`summary.json`)

```json
[
  {
    "patient_id": "751",
    "heart_baseline_silhouette_2d": 0.3245,
    "heart_proposed_silhouette_2d": 0.4512,
    "heart_improvement_2d": 0.1267,
    "heart_baseline_silhouette_3d": 0.3456,
    "heart_proposed_silhouette_3d": 0.4723,
    "heart_improvement_3d": 0.1267,
    "coronary_baseline_silhouette_2d": 0.2134,
    "coronary_proposed_silhouette_2d": 0.3567,
    "coronary_improvement_2d": 0.1433,
    "coronary_baseline_silhouette_3d": 0.2345,
    "coronary_proposed_silhouette_3d": 0.3778,
    "coronary_improvement_3d": 0.1433
  },
  ...
]
```

**Silhouette Score 해석:**
- 범위: -1 ~ 1
- 높을수록 좋음 (클러스터가 잘 분리됨)
- Proposed > Baseline → SPADE가 해부학적 구조를 더 잘 학습

## ⚙️ 커맨드 라인 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--baseline_ckpt` | str | **필수** | Baseline 모델 체크포인트 경로 |
| `--proposed_ckpt` | str | **필수** | Proposed 모델 체크포인트 경로 |
| `--data_dir` | str | data/imageCAS/test | 테스트 데이터 디렉토리 |
| `--num_patients` | int | 5 | 분석할 환자 수 (`--patient_ids` 미지정 시) |
| `--patient_ids` 🆕 | str | None | 특정 환자 ID들 (쉼표 구분, 예: "880,980,1000") |
| `--output_dir` | str | result/experiments/feature_visualization | 결과 저장 디렉토리 |
| `--gpu_number` | int | 0 | 사용할 GPU 번호 |
| `--num_samples` | int | 1000 | 각 해부학적 위치당 샘플 수 (t-SNE용) |
| `--perplexity` | int | 30 | t-SNE perplexity 파라미터 |

## 🔬 분석 방법

### 1. Feature Extraction
- **Hook 메커니즘**: `register_forward_hook`을 사용하여 decoder의 각 layer output 추출
- **Target Layers**: `up_layers.0`, `up_layers.1`, `up_layers.2`
- **Sliding Window Inference**: Patch 단위로 inference하여 전체 볼륨의 feature 추출

### 2. Feature Sampling
- 해부학적 segmentation map을 기준으로 각 구조별 voxel 샘플링
- Background (label 0) 제외
- 각 구조당 최대 1000개 voxel 샘플링

### 3. t-SNE Dimensionality Reduction
1. **PCA 전처리**: 50차원으로 먼저 축소 (속도 향상)
2. **t-SNE**: 2차원으로 최종 축소
3. **Silhouette Score**: 클러스터링 품질 측정

## 📈 결과 해석

### Feature Map 분석
- **Difference가 큰 영역**: SPADE가 크게 영향을 주는 부분
- **해부학적 경계 부근**: 주로 차이가 크게 나타남
- **균일한 영역**: SPADE 영향이 상대적으로 작음

### t-SNE 분석
- **잘 분리된 클러스터**: 해부학적 구조별로 feature가 명확히 구분됨
- **Silhouette Score 향상**: SPADE가 해부학적 정보를 더 잘 활용함을 의미
- **겹치는 영역**: 해부학적으로 인접한 구조 (예: ventricle과 atrium)

## 💡 사용 팁

### 1. 빠른 테스트
환자 1명으로 빠르게 결과 확인:
```bash
bash src/experiments/run_feature_visualization.sh \
    baseline.ckpt \
    proposed.ckpt \
    data/imageCAS/test \
    1 \
    0
```

### 2. 특정 환자만 선택 🆕
관심있는 환자 ID들만 분석:
```bash
python src/experiments/feature_visualization.py \
    --baseline_ckpt result/SegResNet_DiceFocalLoss/final_model.ckpt \
    --proposed_ckpt result/proposed_SegResNet_dstMap_DiceFocalLoss/final_model.ckpt \
    --patient_ids "880,980" \
    --gpu_number 0
```

### 3. 고해상도 t-SNE
더 세밀한 클러스터링이 필요한 경우:
```bash
python src/experiments/feature_visualization.py \
    --baseline_ckpt baseline.ckpt \
    --proposed_ckpt proposed.ckpt \
    --num_samples 5000 \
    --perplexity 50
```

### 4. 특정 decoder layer만 분석
`feature_visualization.py`의 171번 줄 수정:
```python
# 원래
layer_names = ['up_layers.0', 'up_layers.1', 'up_layers.2']

# 마지막 layer만
layer_names = ['up_layers.2']
```

## 📝 주의사항

1. **체크포인트 호환성**: 
   - Baseline은 `src/models/model/segresnet.py` 기반 모델이어야 함
   - Proposed는 `src/models/proposed/segresnet.py` 기반 모델이어야 함

2. **데이터 요구사항**:
   - `img.nii.gz`: 입력 이미지
   - `heart_combined.nii.gz`: 해부학적 segmentation map (필수!)

3. **메모리 사용량**:
   - Feature 추출 시 GPU 메모리 많이 사용
   - t-SNE 계산 시 CPU 메모리 많이 사용
   - `num_samples`를 줄이면 메모리 절약 가능

4. **실행 시간**:
   - 환자 1명당 약 5~10분 소요
   - t-SNE 계산이 가장 오래 걸림 (환자당 2~3분)

## 🔍 문제 해결

### Q: "Checkpoint not found" 오류
**A**: 체크포인트 경로를 확인하세요. Lightning 체크포인트는 자동으로 처리됩니다.

### Q: Feature map이 비어있음
**A**: Hook이 제대로 등록되지 않았을 수 있습니다. Layer 이름을 확인하세요.

### Q: t-SNE가 너무 느림
**A**: `--num_samples`를 500으로 줄이거나 `--perplexity`를 20으로 낮추세요.

### Q: Silhouette score가 음수
**A**: 클러스터가 잘 분리되지 않은 것입니다. Segmentation map의 품질을 확인하세요.

## 📚 참고 문헌

- **t-SNE**: van der Maaten & Hinton, 2008
- **Silhouette Score**: Rousseeuw, 1987
- **SPADE**: Park et al., CVPR 2019

---

**작성일**: 2026-01-27  
**작성자**: Feature Visualization Script
