"""
통계적 유의성 검증 스크립트
Proposed 모델과 Baseline 모델의 성능 차이가 통계적으로 유의미한지 검증합니다.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import argparse
class StatisticalSignificanceTest:
    """두 모델 간의 통계적 유의성을 검증하는 클래스"""
    
    def __init__(self, proposed_path: str, baseline_path: str):
        """
        Args:
            proposed_path: Proposed 모델의 결과 CSV 파일 경로
            baseline_path: Baseline 모델의 결과 CSV 파일 경로
        """
        self.proposed_path = Path(proposed_path)
        self.baseline_path = Path(baseline_path)
        self.proposed_df = None
        self.baseline_df = None
        self.metrics = []
        self.results = {}
        
    def _read_csv_with_encoding(self, file_path: Path) -> pd.DataFrame:
        """여러 인코딩을 시도하여 CSV 파일을 읽습니다."""
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # 다른 오류는 다시 발생시킴
                raise
        
        # 모든 인코딩 실패 시
        raise ValueError(f"CSV 파일을 읽을 수 없습니다. 시도한 인코딩: {encodings}")
    
    def load_data(self) -> None:
        """CSV 파일을 로드하고 데이터를 준비합니다."""
        print(f"Loading data from:")
        print(f"  Proposed: {self.proposed_path}")
        print(f"  Baseline: {self.baseline_path}")
        
        # CSV 파일 로드 (여러 인코딩 시도)
        self.proposed_df = self._read_csv_with_encoding(self.proposed_path)
        self.baseline_df = self._read_csv_with_encoding(self.baseline_path)
        
        # 마지막 행(평균±표준편차) 제거
        self.proposed_df = self.proposed_df[self.proposed_df['patient_id'] != 'AVG ± STD'].copy()
        self.baseline_df = self.baseline_df[self.baseline_df['patient_id'] != 'AVG ± STD'].copy()
        
        # patient_id를 기준으로 정렬
        self.proposed_df = self.proposed_df.sort_values('patient_id').reset_index(drop=True)
        self.baseline_df = self.baseline_df.sort_values('patient_id').reset_index(drop=True)
        
        # 메트릭 목록 추출 (patient_id 제외)
        self.metrics = [col for col in self.proposed_df.columns if col != 'patient_id']
        
        # 데이터 타입을 숫자로 변환
        for col in self.metrics:
            self.proposed_df[col] = pd.to_numeric(self.proposed_df[col], errors='coerce')
            self.baseline_df[col] = pd.to_numeric(self.baseline_df[col], errors='coerce')
        
        print(f"\n데이터 로드 완료:")
        print(f"  환자 수: {len(self.proposed_df)}")
        print(f"  메트릭 수: {len(self.metrics)}")
        print(f"  메트릭: {self.metrics}")
        
    def perform_paired_ttest(self) -> Dict[str, Dict[str, float]]:
        """
        각 메트릭에 대해 paired t-test를 수행합니다.
        
        Returns:
            메트릭별 통계 결과를 담은 딕셔너리
        """
        print("\n" + "="*80)
        print("Paired T-Test 수행 중...")
        print("="*80)
        
        results = {}
        
        for metric in self.metrics:
            proposed_values = self.proposed_df[metric].values
            baseline_values = self.baseline_df[metric].values
            
            # NaN 값이 있는 경우 제외
            mask = ~(np.isnan(proposed_values) | np.isnan(baseline_values))
            proposed_clean = proposed_values[mask]
            baseline_clean = baseline_values[mask]
            
            # Paired t-test 수행
            t_statistic, p_value = stats.ttest_rel(proposed_clean, baseline_clean)
            
            # 기본 통계량 계산
            proposed_mean = np.mean(proposed_clean)
            baseline_mean = np.mean(baseline_clean)
            proposed_std = np.std(proposed_clean, ddof=1)
            baseline_std = np.std(baseline_clean, ddof=1)
            difference = proposed_mean - baseline_mean
            percent_change = (difference / baseline_mean) * 100 if baseline_mean != 0 else 0
            
            # Effect size (Cohen's d) 계산
            pooled_std = np.sqrt((proposed_std**2 + baseline_std**2) / 2)
            cohens_d = difference / pooled_std if pooled_std != 0 else 0
            
            results[metric] = {
                'proposed_mean': proposed_mean,
                'proposed_std': proposed_std,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'difference': difference,
                'percent_change': percent_change,
                't_statistic': t_statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_samples': len(proposed_clean),
                'significant': p_value < 0.05
            }
        
        self.results = results
        return results
    
    def perform_wilcoxon_test(self) -> Dict[str, Dict[str, float]]:
        """
        각 메트릭에 대해 Wilcoxon signed-rank test를 수행합니다.
        (비모수 검정 - 정규분포를 가정하지 않음)
        
        Returns:
            메트릭별 통계 결과를 담은 딕셔너리
        """
        print("\n" + "="*80)
        print("Wilcoxon Signed-Rank Test 수행 중...")
        print("="*80)
        
        wilcoxon_results = {}
        
        for metric in self.metrics:
            proposed_values = self.proposed_df[metric].values
            baseline_values = self.baseline_df[metric].values
            
            # NaN 값이 있는 경우 제외
            mask = ~(np.isnan(proposed_values) | np.isnan(baseline_values))
            proposed_clean = proposed_values[mask]
            baseline_clean = baseline_values[mask]
            
            # Wilcoxon signed-rank test 수행
            try:
                statistic, p_value = stats.wilcoxon(proposed_clean, baseline_clean)
                
                wilcoxon_results[metric] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                print(f"  Warning: {metric}에서 Wilcoxon 테스트 실패 - {e}")
                wilcoxon_results[metric] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False
                }
        
        return wilcoxon_results
    
    def print_results(self) -> None:
        """결과를 콘솔에 출력합니다."""
        print("\n" + "="*80)
        print("통계적 유의성 검증 결과")
        print("="*80)
        
        # Wilcoxon test도 수행
        wilcoxon_results = self.perform_wilcoxon_test()
        
        print(f"\n{'Metric':<25} {'Proposed':<15} {'Baseline':<15} {'Diff':<12} {'Change':<10} {'p-value':<12} {'Sig':<5} {'Cohens d':<10}")
        print("-" * 125)
        
        for metric in self.metrics:
            r = self.results[metric]
            sig_marker = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
            
            print(f"{metric:<25} "
                  f"{r['proposed_mean']:>6.4f}±{r['proposed_std']:<5.4f} "
                  f"{r['baseline_mean']:>6.4f}±{r['baseline_std']:<5.4f} "
                  f"{r['difference']:>+10.4f} "
                  f"{r['percent_change']:>+8.2f}% "
                  f"{r['p_value']:>10.4f}  "
                  f"{sig_marker:<5} "
                  f"{r['cohens_d']:>8.4f}")
        
        print("\n" + "-" * 125)
        print("유의수준: * p<0.05, ** p<0.01, *** p<0.001")
        print("\nCohen's d 해석: |d|<0.2(작음), 0.2≤|d|<0.5(중간), 0.5≤|d|<0.8(큼), |d|≥0.8(매우 큼)")
        
        # 유의미한 개선이 있는 메트릭 요약
        print("\n" + "="*80)
        print("유의미한 개선이 있는 메트릭 (p < 0.05):")
        print("="*80)
        
        improved_metrics = [m for m in self.metrics if self.results[m]['significant'] and self.results[m]['difference'] > 0]
        degraded_metrics = [m for m in self.metrics if self.results[m]['significant'] and self.results[m]['difference'] < 0]
        
        if improved_metrics:
            print("\n[개선됨]")
            for metric in improved_metrics:
                r = self.results[metric]
                print(f"  • {metric}: {r['percent_change']:+.2f}% (p={r['p_value']:.4f}, d={r['cohens_d']:.4f})")
        
        if degraded_metrics:
            print("\n[저하됨]")
            for metric in degraded_metrics:
                r = self.results[metric]
                print(f"  • {metric}: {r['percent_change']:+.2f}% (p={r['p_value']:.4f}, d={r['cohens_d']:.4f})")
        
        if not improved_metrics and not degraded_metrics:
            print("  통계적으로 유의미한 차이를 보이는 메트릭이 없습니다.")
        
        # Wilcoxon test 결과 요약
        print("\n" + "="*80)
        print("Wilcoxon Signed-Rank Test 결과 (비모수 검정):")
        print("="*80)
        print(f"\n{'Metric':<25} {'p-value':<12} {'Significant':<15}")
        print("-" * 55)
        
        for metric in self.metrics:
            w = wilcoxon_results[metric]
            sig_marker = "Yes ***" if w['p_value'] < 0.001 else "Yes **" if w['p_value'] < 0.01 else "Yes *" if w['p_value'] < 0.05 else "No"
            print(f"{metric:<25} {w['p_value']:>10.4f}  {sig_marker:<15}")
    
    def save_results_to_csv(self, output_path: str) -> None:
        """결과를 CSV 파일로 저장합니다."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과를 DataFrame으로 변환
        results_list = []
        for metric, values in self.results.items():
            results_list.append({
                'metric': metric,
                'proposed_mean': values['proposed_mean'],
                'proposed_std': values['proposed_std'],
                'baseline_mean': values['baseline_mean'],
                'baseline_std': values['baseline_std'],
                'difference': values['difference'],
                'percent_change': values['percent_change'],
                't_statistic': values['t_statistic'],
                'p_value': values['p_value'],
                'cohens_d': values['cohens_d'],
                'significant': values['significant'],
                'n_samples': values['n_samples']
            })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_path, index=False)
        print(f"\n결과가 저장되었습니다: {output_path}")
    
    def plot_results(self, output_dir: str) -> None:
        """결과를 시각화하여 저장합니다."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. P-value 바 차트
        self._plot_pvalue_bar(output_dir)
        
        # 2. 메트릭별 박스플롯 비교 (통합)
        self._plot_boxplots_combined(output_dir)
        
        # 3. 메트릭별 박스플롯 비교 (개별)
        self._plot_boxplots_individual(output_dir)
        
        # 4. 차이 분포 히스토그램
        self._plot_difference_distributions(output_dir)
        
        print(f"\n시각화 결과가 저장되었습니다: {output_dir}")
    
    def _plot_pvalue_bar(self, output_dir: Path) -> None:
        """P-value를 바 차트로 시각화합니다."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        metrics = list(self.results.keys())
        p_values = [self.results[m]['p_value'] for m in metrics]
        colors = ['green' if self.results[m]['significant'] else 'gray' for m in metrics]
        
        bars = ax.bar(range(len(metrics)), p_values, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05 (유의수준)')
        ax.axhline(y=0.01, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label='p = 0.01')
        ax.axhline(y=0.001, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7, label='p = 0.001')
        
        # 각 바 위에 p-value 텍스트 표시
        for i, (metric, p_val) in enumerate(zip(metrics, p_values)):
            p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
            ax.text(i, p_val * 2, p_text, ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('P-value (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Statistical Significance Test Results (Paired T-Test)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'p_values_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _format_pvalue(self, p_value: float) -> str:
        """p-value를 보기 좋게 포맷팅합니다."""
        if p_value < 0.001:
            return "< 0.001"
        else:
            return f"= {p_value:.3f}"
    
    def _plot_boxplots_combined(self, output_dir: Path) -> None:
        """메트릭별 박스플롯을 통합 그래프로 생성합니다."""
        n_metrics = len(self.metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            
            proposed_values = self.proposed_df[metric].dropna()
            baseline_values = self.baseline_df[metric].dropna()
            
            data = pd.DataFrame({
                'Proposed': proposed_values,
                'Baseline': baseline_values
            })
            
            bp = ax.boxplot([data['Baseline'], data['Proposed']], 
                           labels=['Baseline', 'Proposed'],
                           patch_artist=True,
                           showmeans=True)
            
            # 색상 설정
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            
            # 유의성 표시
            r = self.results[metric]
            if r['significant']:
                sig_marker = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*"
                y_max = max(data.max())
                y_min = min(data.min())
                y_range = y_max - y_min
                ax.text(1.5, y_max + y_range * 0.08, sig_marker, ha='center', fontsize=16, fontweight='bold')
            
            # p-value 포맷팅
            p_text = self._format_pvalue(r['p_value'])
            ax.set_title(f"{metric}\n(p {p_text}, Δ={r['percent_change']:+.2f}%)", 
                        fontsize=10, pad=15)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        # 빈 subplot 제거
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Comparison of Metrics: Baseline vs Proposed', 
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_dir / 'metrics_boxplots_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots_individual(self, output_dir: Path) -> None:
        """각 메트릭별로 개별 박스플롯을 생성합니다."""
        boxplots_dir = output_dir / 'individual_boxplots'
        boxplots_dir.mkdir(parents=True, exist_ok=True)
        
        for metric in self.metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            proposed_values = self.proposed_df[metric].dropna()
            baseline_values = self.baseline_df[metric].dropna()
            
            data = pd.DataFrame({
                'Proposed': proposed_values,
                'Baseline': baseline_values
            })
            
            bp = ax.boxplot([data['Baseline'], data['Proposed']], 
                           labels=['Baseline', 'Proposed'],
                           patch_artist=True,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            
            # 색상 설정
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightgreen')
            bp['boxes'][1].set_alpha(0.7)
            
            # 통계 정보
            r = self.results[metric]
            
            # y축 범위를 먼저 계산
            y_max = max(data.max())
            y_min = min(data.min())
            y_range = y_max - y_min
            
            # 유의성 표시
            if r['significant']:
                sig_marker = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*"
                
                # 유의성 브래킷 그리기
                bracket_y = y_max + y_range * 0.08
                ax.plot([1, 2], [bracket_y, bracket_y], 'k-', linewidth=1.5)
                ax.plot([1, 1], [bracket_y - y_range * 0.015, bracket_y], 'k-', linewidth=1.5)
                ax.plot([2, 2], [bracket_y - y_range * 0.015, bracket_y], 'k-', linewidth=1.5)
                ax.text(1.5, bracket_y + y_range * 0.03, sig_marker, ha='center', fontsize=20, fontweight='bold')
            
            # p-value 포맷팅
            p_text = self._format_pvalue(r['p_value'])
            
            # 제목 설정 (메인 제목과 subtitle을 그래프 위쪽에 배치)
            title_text = f"{metric}"
            subtitle_text = f"p {p_text}, Δ = {r['percent_change']:+.2f}%, Cohen's d = {r['cohens_d']:.3f}"
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=40)
            # subtitle을 그래프 바로 위쪽에 배치 (*** 표시 아래)
            ax.text(0.5, 1.02, subtitle_text, transform=ax.transAxes, 
                   ha='center', fontsize=10, style='italic')
            
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # y축 범위 조정 (유의성 표시와 제목을 위한 충분한 여유 공간)
            ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.2)
            
            plt.tight_layout()
            
            # 파일명을 안전하게 생성
            safe_metric_name = metric.replace('_', '-')
            plt.savefig(boxplots_dir / f'{safe_metric_name}_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  개별 박스플롯 {len(self.metrics)}개 생성 완료: {boxplots_dir}")
    
    def _plot_difference_distributions(self, output_dir: Path) -> None:
        """각 메트릭의 차이 분포를 히스토그램으로 시각화합니다."""
        n_metrics = len(self.metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            
            proposed_values = self.proposed_df[metric].values
            baseline_values = self.baseline_df[metric].values
            
            # NaN 제거
            mask = ~(np.isnan(proposed_values) | np.isnan(baseline_values))
            differences = proposed_values[mask] - baseline_values[mask]
            
            ax.hist(differences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
            ax.axvline(x=np.mean(differences), color='green', linestyle='-', linewidth=2, 
                      label=f'Mean: {np.mean(differences):.4f}')
            
            r = self.results[metric]
            sig_text = "Significant" if r['significant'] else "Not significant"
            p_text = self._format_pvalue(r['p_value'])
            
            ax.set_title(f"{metric}\n{sig_text} (p {p_text})", fontsize=10, pad=10)
            ax.set_xlabel('Difference (Proposed - Baseline)', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        # 빈 subplot 제거
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Distribution of Differences (Proposed - Baseline)', 
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_dir / 'difference_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_model(project_root: Path, model_name: str) -> None:
    """
    특정 모델에 대해 통계적 유의성 검증을 수행합니다.
    
    Args:
        project_root: 프로젝트 루트 경로
        model_name: 모델 이름 (예: 'SegResNet', 'VNet', 'UNETR')
    """
    print("\n" + "="*80)
    print(f"모델 분석 시작: {model_name}")
    print("="*80)
    
    # 입력 파일 경로
    proposed_path = project_root / f"result/proposed_{model_name}_dstMap/test/test_result.csv"
    baseline_path = project_root / f"result/{model_name}/test/test_result.csv"
    
    # 파일 존재 여부 확인
    if not proposed_path.exists():
        print(f"  경고: Proposed 모델 파일을 찾을 수 없습니다: {proposed_path}")
        return
    if not baseline_path.exists():
        print(f"  경고: Baseline 모델 파일을 찾을 수 없습니다: {baseline_path}")
        return
    
    # 출력 디렉토리 (모델별로 분리)
    output_dir = project_root / f"result/experiments/statistical_significance/proposed_{model_name}_dstMap"
    
    # 통계적 유의성 검증 수행
    tester = StatisticalSignificanceTest(proposed_path, baseline_path)
    tester.load_data()
    tester.perform_paired_ttest()
    tester.print_results()
    
    # 결과 저장
    tester.save_results_to_csv(output_dir / "statistical_test_results.csv")
    tester.plot_results(output_dir)
    
    print(f"\n{model_name} 모델 분석 완료!")
    print(f"결과 저장 위치: {output_dir}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Proposed 모델과 Baseline 모델 간의 통계적 유의성 검증을 수행합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 모델 분석
  python statistical_significance_test.py --models SegResNet
  
  # 여러 모델 동시 분석
  python statistical_significance_test.py --models SegResNet VNet UNETR
  
  # 모든 기본 모델 분석
  python statistical_significance_test.py --models SegResNet VNet UNETR nnFormer CSNet3D
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['SegResNet'],
        help='분석할 모델 이름들 (기본값: SegResNet). 예: --models SegResNet VNet UNETR'
    )
    
    args = parser.parse_args()
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent.parent
    
    print("\n" + "="*80)
    print("통계적 유의성 검증 시작")
    print("="*80)
    print(f"분석할 모델: {', '.join(args.models)}")
    print(f"프로젝트 루트: {project_root}")
    
    # 각 모델에 대해 분석 수행
    successful_models = []
    failed_models = []
    
    for model_name in args.models:
        try:
            analyze_model(project_root, model_name)
            successful_models.append(model_name)
        except Exception as e:
            print(f"\n오류: {model_name} 모델 분석 중 오류 발생 - {e}")
            failed_models.append(model_name)
    
    # 최종 요약
    print("\n" + "="*80)
    print("전체 분석 요약")
    print("="*80)
    print(f"성공한 모델 ({len(successful_models)}): {', '.join(successful_models)}")
    if failed_models:
        print(f"실패한 모델 ({len(failed_models)}): {', '.join(failed_models)}")
    print("\n모든 분석이 완료되었습니다!")
    print("="*80)


if __name__ == "__main__":
    main()
