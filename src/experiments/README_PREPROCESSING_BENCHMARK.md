# TotalSegmentator Heart 전처리 벤치마크 도구

TotalSegmentator를 사용한 Heart ROI (관상동맥 + 심장) 분할 전처리 시간을 측정합니다.

## 측정 항목

1. **Heart ROI 분할 시간** (초)
   - Coronary Arteries (관상동맥)
   - Heart Chambers (심장방/심실)

2. **Peak GPU Memory** (GB)
   - Heart 분할 시 최대 GPU 메모리 사용량

3. **성공률** (%)

## 사용법

### 1. Python 스크립트 직접 실행

```bash
# Heart ROI 벤치마크 (5명 환자, GPU 4번)
python src/experiments/preprocessing_time.py \
    --input_dir data/imageCAS/test \
    --output_root result/experiments/preprocessing \
    --num_patients 5 \
    --gpu_number 4

# 더 많은 환자로 측정
python src/experiments/preprocessing_time.py \
    --input_dir data/imageCAS/test \
    --output_root result/experiments/preprocessing \
    --num_patients 10 \
    --gpu_number 0
```

#### 옵션:

- `--input_dir`: 입력 데이터 디렉토리 (환자 폴더들이 있는 곳, 기본값: `data/imageCAS/test`)
- `--output_root`: 출력 디렉토리 (기본값: `result/experiments/preprocessing`)
- `--num_patients`: 측정할 환자 수 (기본값: 5)
- `--gpu_number`: 사용할 GPU 번호 (기본값: 0)

### 2. 쉘 스크립트로 실행

```bash
# 기본 설정 (5명, GPU 4번)
bash src/experiments/run_preprocessing_time.sh

# 커스텀 설정
bash src/experiments/run_preprocessing_time.sh NUM_PATIENTS GPU_NUMBER

# 예시:
# 10명 환자, GPU 0번
bash src/experiments/run_preprocessing_time.sh 10 0

# 3명 환자, GPU 2번
bash src/experiments/run_preprocessing_time.sh 3 2
```

## 출력 결과

### 1. 콘솔 출력

```
======================================================================
TOTALSEGMENTATOR HEART PREPROCESSING BENCHMARK
======================================================================
Input directory: data/imageCAS/test
Output directory: result/experiments/preprocessing
Number of patients: 5
GPU number: 4
Selected ROI: Heart (Coronary Arteries + Heart Chambers)
======================================================================

============================================================
Processing patient: 25
============================================================
...

======================================================================
BENCHMARK RESULTS SUMMARY
======================================================================

HEART:
  Time: 45.32 ± 3.21 seconds
  Mean GPU Memory: 2.4567 GB
  Max GPU Memory: 2.8901 GB
  Success Rate: 100.0%

TOTAL:
  Time: 45.32 ± 3.21 seconds
  Mean GPU Memory: 2.4567 GB
  Max GPU Memory: 2.8901 GB
  Success Rate: 100.0%

======================================================================
Detailed results saved to: result/experiments/preprocessing/preprocessing_benchmark_results.csv
======================================================================
```

### 2. CSV 파일

`result/experiments/preprocessing/preprocessing_benchmark_results.csv`:

| patient_id | heart_time_s | heart_memory_gb | heart_success | total_time_s | total_max_memory_gb |
|------------|--------------|-----------------|---------------|--------------|---------------------|
| 25         | 45.23        | 2.4567          | True          | 45.23        | 2.8901              |
| 50         | 46.78        | 2.5678          | True          | 46.78        | 2.9123              |
| ...        | ...          | ...             | ...           | ...          | ...                 |
| MEAN ± STD | 45.32 ± 3.21 | 2.4567          | 100.0%        | 45.32 ± 3.21 | 2.8901              |

## 논문/보고서 작성 예시

### Table: TotalSegmentator Heart 전처리 시간

```markdown
| Process          | Time (s)       | GPU Memory (GB) | Success Rate |
|------------------|----------------|-----------------|--------------|
| Heart ROI        | 45.32 ± 3.21   | 2.89 (max)      | 100%         |
```

### 텍스트 설명

```
Preprocessing: Heart ROI segmentation (including coronary arteries and heart 
chambers) was performed using TotalSegmentator on an NVIDIA RTX A6000 GPU. 
The average preprocessing time was 45.32 ± 3.21 seconds per patient, with a 
peak GPU memory usage of 2.89 GB.
```

또는

```
Data Preparation: We employed TotalSegmentator to extract anatomical priors 
for our proposed guided segmentation network. Specifically, the heart region 
(coronary arteries and cardiac chambers) was automatically segmented from each 
CT volume, requiring an average of 45.32 seconds per patient on an NVIDIA RTX 
A6000 GPU.
```

## 주의사항

1. **GPU 메모리**: TotalSegmentator는 상당한 GPU 메모리를 사용하므로 충분한 VRAM이 필요합니다.

2. **디스크 공간**: 각 환자마다 여러 분할 결과 파일이 생성되므로 충분한 디스크 공간이 필요합니다.

3. **데이터 경로**: `--input_dir`에 지정된 디렉토리에 환자 폴더들이 있어야 하며, 각 폴더 안에 `img.nii.gz` 파일이 있어야 합니다.

4. **환자 수**: 첫 실행 시에는 적은 수(3-5명)로 테스트한 후 늘리는 것을 권장합니다.

5. **결과 파일 정리**: 벤치마크 후 디스크 공간 절약을 위해 출력 폴더를 삭제할 수 있습니다.

## 문제 해결

### CUDA Out of Memory

TotalSegmentator의 내부 설정을 조정하거나 더 작은 배치로 실행하세요.

### 파일을 찾을 수 없음

입력 디렉토리 구조를 확인하세요:
```
data/imageCAS_RAS_affine/test/
├── 25/
│   └── img.nii.gz
├── 50/
│   └── img.nii.gz
...
```

### TotalSegmentator 실행 실패

TotalSegmentator가 올바르게 설치되어 있는지 확인하세요:
```bash
pip install TotalSegmentator
```

## 비교 분석

이 벤치마크 결과를 `benchmark_baseline.py`와 `benchmark_proposed.py`의 inference 시간과 비교하여 
전처리 vs 추론 시간의 비율을 분석할 수 있습니다.

예시:
- Heart 전처리 시간: ~45초/환자
- Inference 시간: ~2초/볼륨
- **전처리가 추론보다 ~20배 더 오래 걸림**

이는 전처리를 사전에 수행하는 것이 실시간 응용에서 중요함을 보여줍니다.

### 전체 파이프라인 시간

1. **전처리 (1회, 오프라인)**: ~45초
2. **Proposed 모델 추론**: ~5-6초
3. **Baseline 모델 추론**: ~2초

→ Proposed 모델은 전처리된 anatomical prior를 활용하여 더 나은 성능을 달성하지만, 
   추가 입력 채널로 인해 약간의 추론 시간 증가가 있습니다.
