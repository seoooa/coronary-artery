import numpy as np
import torch
import nibabel as nib
import math
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CoordinatePositionalEmbedding:
    """
    좌표 기반 위치 임베딩 클래스
    의료 영상의 메타데이터에서 추출한 물리적 좌표와 해상도 정보를 활용한 위치 임베딩 생성
    """
    
    def __init__(self, scaling_factor=100.0, angular_frequency=1.0):
        """
        초기화 함수
        
        Parameters:
        -----------
        scaling_factor : float
            임베딩 값을 스케일링하기 위한 계수
        angular_frequency : float
            사인파의 각 주파수
        """
        self.scaling_factor = scaling_factor
        self.angular_frequency = angular_frequency
    
    def load_medical_image(self, image_path):
        """
        의료 영상 파일(NIfTI)을 로드하고 메타데이터 추출
        
        Parameters:
        -----------
        image_path : str
            의료 영상 파일 경로
            
        Returns:
        --------
        tuple
            (영상 데이터, 원점, 픽셀 간격)
        """
        # NIfTI 파일 로드
        nifti_img = nib.load(image_path)
        
        # 영상 데이터, 원점, 픽셀 간격 추출
        image_data = nifti_img.get_fdata()
        affine = nifti_img.affine
        
        # 원점 좌표 추출 (affine 변환 행렬의 마지막 열)
        origin = affine[:3, 3]
        
        # 픽셀 간격 추출 (affine 변환 행렬의 대각선 요소)
        pixel_spacing = np.array([
            np.sqrt(np.sum(affine[:3, 0]**2)),
            np.sqrt(np.sum(affine[:3, 1]**2)),
            np.sqrt(np.sum(affine[:3, 2]**2))
        ])
        
        return image_data, origin, pixel_spacing, affine
    
    def generate_physical_coordinates(self, shape, origin, spacing):
        """
        물리적 좌표 그리드 생성
        
        Parameters:
        -----------
        shape : tuple
            영상 데이터의 형태 (H, W, D)
        origin : numpy.ndarray
            영상의 원점 좌표
        spacing : numpy.ndarray
            픽셀 간격
            
        Returns:
        --------
        tuple
            (x 좌표 그리드, y 좌표 그리드, z 좌표 그리드)
        """
        # 각 차원에 대한 좌표 배열 생성
        x_coords = origin[0] + np.arange(shape[0]) * spacing[0]
        y_coords = origin[1] + np.arange(shape[1]) * spacing[1]
        z_coords = origin[2] + np.arange(shape[2]) * spacing[2]
        
        # 3D 그리드로 확장
        x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        return x_grid, y_grid, z_grid
    
    def calculate_positional_embedding(self, x_grid, y_grid, z_grid):
        """
        좌표 기반 위치 임베딩 계산
        
        Parameters:
        -----------
        x_grid : numpy.ndarray
            x 좌표 그리드
        y_grid : numpy.ndarray
            y 좌표 그리드
        z_grid : numpy.ndarray
            z 좌표 그리드
            
        Returns:
        --------
        numpy.ndarray
            좌표 기반 위치 임베딩
        """
        # 원점으로부터의 유클리드 거리 계산
        distance = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
        
        # 논문에서 제안한 방사형 사인파 구조: sin(ω√(x²+y²+z²))/S
        # positional_embedding = np.sin(self.angular_frequency * distance) / self.scaling_factor
        positional_embedding = distance

        return positional_embedding
    
    def to_sin_cos_embedding(self, positional_embedding, embed_dim=24):
        """
        사인-코사인 위치 임베딩으로 변환
        
        Parameters:
        -----------
        positional_embedding : numpy.ndarray
            기본 위치 임베딩
        embed_dim : int
            임베딩 차원 (6의 배수여야 함)
            
        Returns:
        --------
        numpy.ndarray
            사인-코사인 위치 임베딩
        """
        assert embed_dim % 6 == 0, "임베딩 차원은 6의 배수여야 합니다."
        
        # 텐서로 변환
        ppe_tensor = torch.from_numpy(positional_embedding).float()
        
        # 임베딩 차원 준비
        frequencies = 2 ** torch.arange(-1, embed_dim // 6 - 1).float()
        
        # 복수의 주파수로 사인-코사인 임베딩 계산
        ppe_expanded = ppe_tensor.unsqueeze(-1) * frequencies * (2 * math.pi)
        
        # 사인과 코사인 임베딩 계산
        sin_embedding = torch.sin(ppe_expanded)
        cos_embedding = torch.cos(ppe_expanded)
        
        # 사인과 코사인 임베딩 결합
        sin_cos_embedding = torch.cat([sin_embedding, cos_embedding], dim=-1)
        
        return sin_cos_embedding.numpy()
    
    def generate_embedding(self, image_path, output_path=None, visualize=False, generate_sin_cos=False):
        """
        의료 영상의 메타데이터를 사용하여 위치 임베딩 생성 및 저장
        
        Parameters:
        -----------
        image_path : str
            의료 영상 파일 경로
        output_path : str, optional
            결과 저장 경로
        visualize : bool
            시각화 여부
        generate_sin_cos : bool
            사인-코사인 임베딩 생성 여부
        """
        # 의료 영상 로드
        image_data, origin, pixel_spacing, affine = self.load_medical_image(image_path)
        
        # 물리적 좌표 그리드 생성
        x_grid, y_grid, z_grid = self.generate_physical_coordinates(
            image_data.shape, origin, pixel_spacing
        )
        
        # 좌표 기반 위치 임베딩 계산
        positional_embedding = self.calculate_positional_embedding(x_grid, y_grid, z_grid)
        
        # 사인-코사인 위치 임베딩 변환 (선택적)
        sin_cos_embedding = None
        if generate_sin_cos:
            sin_cos_embedding = self.to_sin_cos_embedding(positional_embedding)
        
        # 결과 저장
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 기본 위치 임베딩 저장
            ppe_nifti = nib.Nifti1Image(positional_embedding, affine)
            nib.save(ppe_nifti, output_dir / "ppe.nii.gz")
            
            # 사인-코사인 위치 임베딩 저장 (선택적)
            if generate_sin_cos:
                np.save(output_dir / "ppe_sin_cos.npy", sin_cos_embedding)
            
            print(f"위치 임베딩 저장 완료: {output_dir}")
            print(f"1. 기본 위치 임베딩: {output_dir}/ppe.nii.gz")
            if generate_sin_cos:
                print(f"2. 사인-코사인 위치 임베딩: {output_dir}/ppe_sin_cos.npy")
        
        # 시각화
        if visualize:
            self._visualize_embedding(image_data, positional_embedding)
        
        return positional_embedding, sin_cos_embedding
    
    def _visualize_embedding(self, image_data, positional_embedding):
        """
        원본 영상과 위치 임베딩 시각화
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            원본 영상 데이터
        positional_embedding : numpy.ndarray
            위치 임베딩
        """
        # 중앙 슬라이스 선택
        slice_idx = image_data.shape[2] // 2
        
        # 2x1 서브플롯 생성
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 원본 영상 표시
        axes[0].imshow(image_data[:, :, slice_idx].T, cmap='gray')
        axes[0].set_title('원본 영상')
        axes[0].axis('off')
        
        # 위치 임베딩 표시
        im = axes[1].imshow(positional_embedding[:, :, slice_idx].T, cmap='viridis')
        axes[1].set_title('좌표 기반 위치 임베딩')
        axes[1].axis('off')
        
        # 컬러바 추가
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

    def process_all_patients(self, base_dir, generate_sin_cos=False):
        """
        모든 환자의 이미지에 대해 위치 임베딩을 생성하고 저장
        이미 ppe.nii.gz가 있는 환자는 건너뜀
        
        Parameters:
        -----------
        base_dir : str
            imageCAS_heart 폴더 경로
        generate_sin_cos : bool
            사인-코사인 임베딩 생성 여부
        """
        # 데이터셋 분할 (train, valid, test)
        splits = ['test']
        
        total_processed = 0
        total_errors = 0
        total_skipped = 0
        
        for split in splits:
            split_dir = Path(base_dir) / split
            if not split_dir.exists():
                print(f"경고: {split_dir} 폴더가 존재하지 않습니다.")
                continue
                
            print(f"\n{split} 데이터셋 처리 중...")
            
            # 환자 폴더 목록
            patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            
            if split == 'train':
                # train 데이터셋을 3개 그룹으로 나누기
                total_patients = len(patient_dirs)
                group1_size = total_patients // 3
                group2_size = (total_patients - group1_size) // 2
                group3_size = total_patients - group1_size - group2_size
                
                groups = [
                    ("group1", patient_dirs[:group1_size]),
                    # ("group2", patient_dirs[group1_size:group1_size + group2_size]),
                    # ("group3", patient_dirs[group1_size + group2_size:])
                ]
                
                for group_name, group_dirs in groups:
                    print(f"\n===== Processing Train {group_name} ({len(group_dirs)} patients) =====")
                    
                    # 각 그룹의 환자들 처리
                    for patient_dir in tqdm(group_dirs, desc=f"Train {group_name} 처리"):
                        # ppe.nii.gz 파일이 이미 존재하는지 확인
                        ppe_path = patient_dir / "ppe.nii.gz"
                        if ppe_path.exists():
                            total_skipped += 1
                            continue
                            
                        img_path = patient_dir / "img.nii.gz"
                        if not img_path.exists():
                            print(f"경고: {img_path}가 존재하지 않습니다.")
                            continue
                        
                        try:
                            # 임베딩 생성 및 저장
                            positional_embedding, sin_cos_embedding = self.generate_embedding(
                                str(img_path),
                                str(patient_dir),
                                visualize=False,
                                generate_sin_cos=generate_sin_cos
                            )
                            total_processed += 1
                            
                        except Exception as e:
                            print(f"오류: {patient_dir.name} 처리 중 오류 발생 - {str(e)}")
                            total_errors += 1
                    
                    print(f"===== Train {group_name} completed =====")
            
            else:
                # valid와 test 데이터셋은 그대로 처리
                for patient_dir in tqdm(patient_dirs, desc=f"{split} 환자 처리"):
                    # ppe.nii.gz 파일이 이미 존재하는지 확인
                    ppe_path = patient_dir / "ppe.nii.gz"
                    if ppe_path.exists():
                        total_skipped += 1
                        continue
                        
                    img_path = patient_dir / "img.nii.gz"
                    if not img_path.exists():
                        print(f"경고: {img_path}가 존재하지 않습니다.")
                        continue
                    
                    try:
                        # 임베딩 생성 및 저장
                        positional_embedding, sin_cos_embedding = self.generate_embedding(
                            str(img_path),
                            str(patient_dir),
                            visualize=False,
                            generate_sin_cos=generate_sin_cos
                        )
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"오류: {patient_dir.name} 처리 중 오류 발생 - {str(e)}")
                        total_errors += 1
        
        print(f"\n처리 완료:")
        print(f"- 성공적으로 처리된 환자 수: {total_processed}")
        print(f"- 건너뛴 환자 수 (이미 ppe.nii.gz 존재): {total_skipped}")
        print(f"- 오류 발생 환자 수: {total_errors}")


# 사용 예시
if __name__ == "__main__":
    # 좌표 기반 위치 임베딩 인스턴스 생성
    ppe = CoordinatePositionalEmbedding(scaling_factor=100.0, angular_frequency=1.0)
    
    # imageCAS_heart 폴더 경로 설정
    base_dir = "/home/seoooa/project/coronary-artery/data/imageCAS_heart"
    
    # sin-cos 임베딩 생성 여부 설정
    generate_sin_cos = False  # True로 설정하면 sin-cos 임베딩도 생성
    
    # 모든 환자 데이터 처리
    ppe.process_all_patients(base_dir, generate_sin_cos=generate_sin_cos)