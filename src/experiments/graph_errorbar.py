import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import os
from pathlib import Path

# --- Data Preparation ---
baseline_dice = 0.7658
baseline_cldice = 0.8155  # baseline clDice 값 (필요시 수정)

# CSV 파일 경로 설정
csv_base_dir = Path("result/experiments/sdm_perturbation")

def load_csv_and_calculate_ci(csv_path, perturbation_level):
    """CSV 파일을 읽어서 실제 샘플 수를 계산하고 98% 신뢰구간을 반환 (Dice와 clDice 모두)"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return None, None, None, None
    
    df = pd.read_csv(csv_path)
    
    # "MEAN ± STD" 행 제외하고 실제 데이터만 추출
    df = df[df['patient_id'] != 'MEAN ± STD']
    
    # Dice와 clDice 점수를 숫자로 변환 (변환 불가능한 행은 NaN으로 처리)
    df['dice'] = pd.to_numeric(df['dice'], errors='coerce')
    df['cldice'] = pd.to_numeric(df['cldice'], errors='coerce')
    
    # NaN이 있는 행 제거
    df = df.dropna(subset=['dice', 'cldice'])
    
    # 실제 샘플 수 계산
    n_samples = len(df)
    
    if n_samples == 0:
        print(f"Warning: No valid data found in {csv_path}")
        return None, None, None, None
    
    # Dice 점수 추출 및 계산
    dice_scores = df['dice'].values
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores, ddof=1)
    se_dice = std_dice / np.sqrt(n_samples)
    ci_98_dice = 2.326 * se_dice
    
    # clDice 점수 추출 및 계산
    cldice_scores = df['cldice'].values
    mean_cldice = np.mean(cldice_scores)
    std_cldice = np.std(cldice_scores, ddof=1)
    se_cldice = std_cldice / np.sqrt(n_samples)
    ci_98_cldice = 2.326 * se_cldice
    
    print(f"  Level {perturbation_level}: n={n_samples}, Dice={mean_dice:.4f}±{ci_98_dice:.6f}, clDice={mean_cldice:.4f}±{ci_98_cldice:.6f}")
    
    return mean_dice, ci_98_dice, mean_cldice, ci_98_cldice

# Robustness - Noise Data
sigma = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
dice_noise = []
ci_98_noise = []
cldice_noise = []
ci_98_cldice_noise = []

print("Loading Noise Data from CSV files...")
for level in sigma:
    csv_path = csv_base_dir / "boundary_noise" / f"level_{level}" / f"detailed_results_boundary_noise_{level}.csv"
    mean_dice, ci_98_dice, mean_cldice, ci_98_cldice = load_csv_and_calculate_ci(csv_path, level)
    if mean_dice is not None:
        dice_noise.append(mean_dice)
        ci_98_noise.append(ci_98_dice)
        cldice_noise.append(mean_cldice)
        ci_98_cldice_noise.append(ci_98_cldice)
    else:
        # CSV 파일이 없으면 에러 발생
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Please ensure the CSV files exist.")

# Robustness - Deformation Data
deformation_mm = [0.0, 1.0, 2.0, 3.0, 4.0]
dice_def = []
ci_98_def = []
cldice_def = []
ci_98_cldice_def = []

print("\nLoading Deformation Data from CSV files...")
for level in deformation_mm:
    csv_path = csv_base_dir / "dilation_erosion" / f"level_{level}" / f"detailed_results_dilation_erosion_{level}.csv"
    mean_dice, ci_98_dice, mean_cldice, ci_98_cldice = load_csv_and_calculate_ci(csv_path, level)
    if mean_dice is not None:
        dice_def.append(mean_dice)
        ci_98_def.append(ci_98_dice)
        cldice_def.append(mean_cldice)
        ci_98_cldice_def.append(ci_98_cldice)
    else:
        # CSV 파일이 없으면 에러 발생
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Please ensure the CSV files exist.")

print("\nData loading completed!")

# --- Plotting ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, (ax_noise, ax_def) = plt.subplots(1, 2, figsize=(16, 5.5))
fig.subplots_adjust(wspace=0.35)  # 서브플롯 간격 조정

# Colors
dice_color = '#1f77b4' # Blue
base_color = 'navy'    # Darker Blue for baseline line
cldice_color = '#7cb342' # Light green for clDice
cldice_base_color = '#558b2f'  # Darker green for baseline clDice

# ==========================================
# Subplot 1: Noise Robustness (a)
# ==========================================
# box_width를 x축 범위에 비례하게 설정 (Noise: 0~10, 상대적 비율 약 3%)
box_width_noise = 0.3  # Noise 그래프용 박스 너비

# 1.1 Discrete Shaded Boxes for Dice 98% Confidence Interval
for i, (x, y, ci) in enumerate(zip(sigma, dice_noise, ci_98_noise)):
    rect = mpatches.Rectangle((x - box_width_noise/2, y - ci), box_width_noise, 2*ci,
                               facecolor=dice_color, alpha=0.2, edgecolor='none', zorder=2)
    ax_noise.add_patch(rect)

# 1.2 Dice Line Chart
l1, = ax_noise.plot(sigma, dice_noise, marker='o', color=dice_color, linewidth=2, label='Proposed Dice', zorder=3)
l1_base = ax_noise.axhline(y=baseline_dice, color=base_color, linestyle=':', linewidth=2, label='Baseline Dice', zorder=3)
ax_noise.text(0.2, baseline_dice - 0.008, f'Baseline DSC: {baseline_dice:.3f}', color=base_color, fontsize=12, fontweight='bold', ha='left', va='top')

# 1.3 clDice on same y-axis with 98% CI
# Discrete Shaded Boxes for clDice 98% Confidence Interval
for i, (x, y, ci) in enumerate(zip(sigma, cldice_noise, ci_98_cldice_noise)):
    rect = mpatches.Rectangle((x - box_width_noise/2, y - ci), box_width_noise, 2*ci,
                               facecolor=cldice_color, alpha=0.2, edgecolor='none', zorder=2)
    ax_noise.add_patch(rect)

l2, = ax_noise.plot(sigma, cldice_noise, marker='s', color=cldice_color, linewidth=2, label='Proposed clDice', zorder=3)
l2_base = ax_noise.axhline(y=baseline_cldice, color=cldice_base_color, linestyle=':', linewidth=2, label='Baseline clDice', zorder=3)
ax_noise.text(0.2, baseline_cldice + 0.008, f'Baseline clDice: {baseline_cldice:.3f}', color=cldice_base_color, fontsize=12, fontweight='bold', ha='left', va='bottom')

ax_noise.set_xlabel('Noise Level ($\sigma$)', fontweight='bold', fontsize=14)
ax_noise.set_ylabel('Dice Score / clDice ↑', color='black', fontweight='bold', fontsize=14)
ax_noise.tick_params(axis='y', labelcolor='black', labelsize=12)
ax_noise.set_ylim(0.70, 0.90)
ax_noise.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
ax_noise.grid(True, linestyle='--', alpha=0.5, zorder=1)
ax_noise.set_title('(a) Robustness to Boundary Noise', fontweight='bold', pad=20, fontsize=16)

# Legend (both metrics) - 2x2 layout: Dice in first column, clDice in second column
# 순서 조정: 열 우선 배치를 위해 [Proposed Dice, Proposed clDice, Baseline Dice, Baseline clDice]
lns = [l1, l2, l1_base, l2_base]
labs = [l.get_label() for l in lns]
ax_noise.legend(lns, labs, loc='upper right', frameon=True, ncol=2, columnspacing=1.0)

# ==========================================
# Subplot 2: Deformation Robustness (b)
# ==========================================
# box_width를 x축 범위에 비례하게 설정 (Deformation: 0~4, Noise와 같은 상대적 비율 3%로 맞춤)
box_width_def = 0.12  # Deformation 그래프용 박스 너비 (0.4 * 3% = 0.12)

# 2.1 Discrete Shaded Boxes for Dice 98% Confidence Interval
for i, (x, y, ci) in enumerate(zip(deformation_mm, dice_def, ci_98_def)):
    rect = mpatches.Rectangle((x - box_width_def/2, y - ci), box_width_def, 2*ci,
                               facecolor=dice_color, alpha=0.2, edgecolor='none', zorder=2)
    ax_def.add_patch(rect)

# 2.2 Dice Line Chart
l3, = ax_def.plot(deformation_mm, dice_def, marker='o', color=dice_color, linewidth=2, label='Proposed Dice', zorder=3)
l3_base = ax_def.axhline(y=baseline_dice, color=base_color, linestyle=':', linewidth=2, label='Baseline Dice', zorder=3)
ax_def.text(0.2, baseline_dice - 0.008, f'Baseline DSC: {baseline_dice:.3f}', color=base_color, fontsize=12, fontweight='bold', ha='left', va='top')

# 2.3 clDice on same y-axis with 98% CI
# Discrete Shaded Boxes for clDice 98% Confidence Interval
for i, (x, y, ci) in enumerate(zip(deformation_mm, cldice_def, ci_98_cldice_def)):
    rect = mpatches.Rectangle((x - box_width_def/2, y - ci), box_width_def, 2*ci,
                               facecolor=cldice_color, alpha=0.2, edgecolor='none', zorder=2)
    ax_def.add_patch(rect)

l4, = ax_def.plot(deformation_mm, cldice_def, marker='s', color=cldice_color, linewidth=2, label='Proposed clDice', zorder=3)
l4_base = ax_def.axhline(y=baseline_cldice, color=cldice_base_color, linestyle=':', linewidth=2, label='Baseline clDice', zorder=3)
ax_def.text(0.2, baseline_cldice + 0.008, f'Baseline clDice: {baseline_cldice:.3f}', color=cldice_base_color, fontsize=12, fontweight='bold', ha='left', va='bottom')

ax_def.set_xlabel('Deformation Magnitude $L$ (mm)', fontweight='bold', fontsize=14)
ax_def.set_ylabel('Dice Score / clDice ↑', color='black', fontweight='bold', fontsize=14)
ax_def.tick_params(axis='y', labelcolor='black', labelsize=12)
ax_def.set_ylim(0.70, 0.90)
ax_def.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
ax_def.grid(True, linestyle='--', alpha=0.5, zorder=1)
ax_def.set_title('(b) Robustness to Spatial Deformation', fontweight='bold', pad=20, fontsize=16)

# Legend (both metrics) - 2x2 layout: Dice in first column, clDice in second column
# 순서 조정: 열 우선 배치를 위해 [Proposed Dice, Proposed clDice, Baseline Dice, Baseline clDice]
lns2 = [l3, l4, l3_base, l4_base]
labs2 = [l.get_label() for l in lns2]
ax_def.legend(lns2, labs2, loc='upper right', frameon=True, ncol=2, columnspacing=1.0)

plt.savefig('robustness_discrete_boxes.png', dpi=300, bbox_inches='tight')
plt.show()
