import numpy as np
import torch
import nibabel as nib
import math
from pathlib import Path
import matplotlib.pyplot as plt

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
        positional_embedding = np.sin(self.angular_frequency * distance) / self.scaling_factor
        
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
    
    def generate_embedding(self, image_path, output_path=None, visualize=False):
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
            
        Returns:
        --------
        tuple
            (위치 임베딩, 사인-코사인 위치 임베딩)
        """
        # 의료 영상 로드
        image_data, origin, pixel_spacing, affine = self.load_medical_image(image_path)
        
        # 물리적 좌표 그리드 생성
        x_grid, y_grid, z_grid = self.generate_physical_coordinates(
            image_data.shape, origin, pixel_spacing
        )
        
        # 좌표 기반 위치 임베딩 계산
        positional_embedding = self.calculate_positional_embedding(x_grid, y_grid, z_grid)
        
        # 사인-코사인 위치 임베딩 변환
        sin_cos_embedding = self.to_sin_cos_embedding(positional_embedding)
        
        # 결과 저장
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 기본 위치 임베딩 저장
            ppe_nifti = nib.Nifti1Image(positional_embedding, affine)
            nib.save(ppe_nifti, output_dir / "positional_embedding.nii.gz")
            
            # 사인-코사인 위치 임베딩 NumPy 파일로 저장
            np.save(output_dir / "sin_cos_embedding.npy", sin_cos_embedding)
            
            print(f"위치 임베딩 저장 완료: {output_dir}")
            print(f"1. 기본 위치 임베딩: {output_dir}/positional_embedding.nii.gz")
            print(f"2. 사인-코사인 위치 임베딩: {output_dir}/sin_cos_embedding.npy")
        
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


# 사용 예시
if __name__ == "__main__":
    # 좌표 기반 위치 임베딩 인스턴스 생성
    ppe = CoordinatePositionalEmbedding(scaling_factor=100.0, angular_frequency=1.0)
    
    # 의료 영상 처리
    image_path = "/home/seoooa/project/coronary-artery/data/imageCAS_heart/train/1/img.nii.gz"  # 실제 경로로 변경
    output_path = "/home/seoooa/project/coronary-artery/data/imageCAS_heart/train/1/ppe_maps.nii.gz"  # 실제 경로로 변경
    
    try:
        # 임베딩만 생성하고 저장
        positional_embedding, sin_cos_embedding = ppe.generate_embedding(
            image_path, output_path, visualize=True
        )
        
        print(f"위치 임베딩 형태: {positional_embedding.shape}")
        print(f"사인-코사인 위치 임베딩 형태: {sin_cos_embedding.shape}")
        
    except Exception as e:
        print(f"오류 발생: {e}")