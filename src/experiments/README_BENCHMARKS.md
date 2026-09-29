# Inference Benchmark Tools

이 디렉토리는 baseline 모델과 proposed 모델의 inference 성능을 측정하는 벤치마크 도구를 포함합니다.

## 측정 항목

각 벤치마크는 다음 항목을 측정합니다:

1. **Inference Time per Patch**: 단일 패치(96×96×96)에 대한 평균 추론 시간 (초)
2. **Inference Time per Volume**: 전체 볼륨에 대한 평균 추론 시간 (초, sliding window 사용)
3. **Peak GPU Memory (Patch)**: 단일 패치 추론 시 최대 GPU 메모리 사용량 (GB)
4. **Peak GPU Memory (Volume)**: 전체 볼륨 추론 시 최대 GPU 메모리 사용량 (GB)

## 파일 설명

- `benchmark_baseline.py`: Baseline 모델 (train.py) 벤치마크
- `benchmark_proposed.py`: Proposed 모델 (proposed_train.py) 벤치마크
- `run_benchmarks.sh`: 모든 벤치마크를 실행하는 쉘 스크립트
- `README_BENCHMARKS.md`: 이 파일

## 사용법

### 1. Baseline 모델 벤치마크

```bash
python src/experiments/benchmark_baseline.py \
    --arch_name SegResNet \
    --checkpoint_path result/SegResNet_DiceFocalLoss/final_model.ckpt \
    --gpu_number 0 \
    --num_runs 10 \
    --warmup_runs 3 \
    --output_dir result/benchmarks
```

#### 옵션:
- `--arch_name`: 모델 아키텍처 (UNet, AttentionUnet, SegResNet, UNETR, SwinUNETR, VNet, DynUNet, CSNet3D, nnFormer)
- `--checkpoint_path`: 체크포인트 파일 경로 (필수)
- `--gpu_number`: 사용할 GPU 번호 (기본값: 0)
- `--num_runs`: 벤치마크 실행 횟수 (기본값: 10)
- `--warmup_runs`: 워밍업 실행 횟수 (기본값: 3)
- `--output_dir`: 결과 저장 디렉토리 (기본값: result/benchmarks)

### 2. Proposed 모델 벤치마크

```bash
python src/experiments/benchmark_proposed.py \
    --arch_name SegResNet \
    --checkpoint_path result/proposed_SegResNet_dstMap_DiceFocalLoss/final_model.ckpt \
    --guide distanceMap \
    --gpu_number 0 \
    --num_runs 10 \
    --warmup_runs 3 \
    --output_dir result/benchmarks
```

#### 추가 옵션:
- `--guide`: Guide 타입 (segMap, distanceMap)

### 3. 쉘 스크립트로 일괄 실행

```bash
# 기본 설정 (SegResNet, GPU 0, 10 runs, 3 warmup runs)
bash src/experiments/run_benchmarks.sh

# 커스텀 설정
bash src/experiments/run_benchmarks.sh ARCH_NAME GPU_NUMBER NUM_RUNS WARMUP_RUNS

# 예시:
# SegResNet, GPU 0, 10 runs, 3 warmup
bash src/experiments/run_benchmarks.sh SegResNet 0 10 3

# UNETR, GPU 1, 20 runs, 5 warmup
bash src/experiments/run_benchmarks.sh UNETR 1 20 5

# SwinUNETR, GPU 2, 50 runs, 10 warmup (정밀 측정)
bash src/experiments/run_benchmarks.sh SwinUNETR 2 50 10
```

#### 파라미터:
1. `ARCH_NAME`: 모델 아키텍처 (기본값: SegResNet)
   - Baseline: UNet, AttentionUnet, SegResNet, UNETR, SwinUNETR, VNet, CSNet3D, nnFormer
   - Proposed: SegResNet, UNETR, SwinUNETR, nnFormer, CSNet3D, AttentionUnet, VNet
2. `GPU_NUMBER`: 사용할 GPU 번호 (기본값: 0)
3. `NUM_RUNS`: 벤치마크 실행 횟수 (기본값: 10)
4. `WARMUP_RUNS`: 워밍업 실행 횟수 (기본값: 3)

## 출력 결과

결과는 CSV 파일로 저장됩니다:

- `result/benchmarks/baseline_{arch_name}_benchmark.csv`
- `result/benchmarks/proposed_{arch_name}_{guide}_benchmark.csv`

### CSV 파일 내용:

| 항목 | 설명 |
|------|------|
| Architecture | 모델 아키텍처 이름 |
| Model Type | 모델 타입 (Baseline / Proposed) |
| Checkpoint | 체크포인트 경로 |
| Patch Inference Time (s) | 패치 추론 시간 ± 표준편차 |
| Volume Inference Time (s) | 볼륨 추론 시간 ± 표준편차 |
| Patch Peak GPU Memory (GB) | 패치 최대 GPU 메모리 |
| Volume Peak GPU Memory (GB) | 볼륨 최대 GPU 메모리 |
| Num Runs | 벤치마크 실행 횟수 |
| Warmup Runs | 워밍업 실행 횟수 |
| Volume Shape | 입력 볼륨 크기 |
| Patch Size | 패치 크기 |
| Guide Type | (Proposed만) Guide 타입 |

## 예제 워크플로우

### 여러 모델 비교하기

#### 방법 1: 쉘 스크립트 반복 사용

```bash
# 여러 아키텍처 벤치마크
for arch in SegResNet UNETR SwinUNETR; do
    echo "Processing $arch..."
    bash src/experiments/run_benchmarks.sh $arch 0 10 3
done
```

#### 방법 2: 개별 Python 스크립트 직접 호출

```bash
# Baseline models
for arch in SegResNet UNETR SwinUNETR; do
    python src/experiments/benchmark_baseline.py \
        --arch_name $arch \
        --checkpoint_path result/${arch}/final_model.ckpt \
        --gpu_number 0
done

# Proposed models (distanceMap)
for arch in SegResNet UNETR SwinUNETR; do
    python src/experiments/benchmark_proposed.py \
        --arch_name $arch \
        --checkpoint_path result/proposed_${arch}_dstMap/final_model.ckpt \
        --guide distanceMap \
        --gpu_number 0
done
```

### 결과 취합하기

모든 벤치마크를 실행한 후, `result/benchmarks/` 디렉토리에 있는 CSV 파일들을 하나로 병합하여 비교할 수 있습니다.

```python
import pandas as pd
from pathlib import Path

# 모든 CSV 파일 읽기
benchmark_dir = Path("result/benchmarks")
csv_files = list(benchmark_dir.glob("*.csv"))

# 데이터프레임 리스트로 읽기
dfs = [pd.read_csv(f) for f in csv_files]

# 병합
combined_df = pd.concat(dfs, ignore_index=True)

# 저장
combined_df.to_csv(benchmark_dir / "all_benchmarks.csv", index=False)

# 출력
print(combined_df.to_string())
```

## 주의사항

1. **GPU 메모리**: 벤치마크 실행 전에 GPU 메모리가 충분한지 확인하세요.
2. **체크포인트 경로**: 체크포인트 파일이 존재하는지 확인하세요.
3. **데이터 경로**: `data/imageCAS` 디렉토리에 테스트 데이터가 있어야 합니다.
4. **재현성**: 더 정확한 측정을 위해 `num_runs`와 `warmup_runs`를 늘릴 수 있습니다.
5. **Background 프로세스**: 벤치마크 실행 중에는 다른 GPU 프로세스를 종료하는 것이 좋습니다.

## 문제 해결

### CUDA Out of Memory
- `sw_batch_size`를 줄이거나 더 작은 `patch_size`를 사용하세요.
- 코드에서 `benchmark_baseline.py`와 `benchmark_proposed.py`의 `sw_batch_size` 값을 조정할 수 있습니다.

### 체크포인트 로드 오류
- 체크포인트 경로가 올바른지 확인하세요.
- 아키텍처 이름이 체크포인트와 일치하는지 확인하세요.

### 데이터 로드 오류
- `data/imageCAS` 디렉토리 구조가 올바른지 확인하세요.
- DataModule의 `prepare_data()` 메서드가 올바르게 실행되는지 확인하세요.
