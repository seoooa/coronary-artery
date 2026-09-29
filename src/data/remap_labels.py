import os
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def remap_labels(data):
    """
    라벨 값을 0,1,4,5,6,7 에서 0,1,2,3,4,5 로 리매핑
    """
    # 새로운 배열 생성
    remapped_data = np.zeros_like(data)
    
    # 매핑 정의
    label_mapping = {
        0: 0,
        1: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5
    }
    
    # 각 라벨에 대해 리매핑 수행
    for old_label, new_label in label_mapping.items():
        remapped_data[data == old_label] = new_label
    
    return remapped_data


def process_file(file_path):
    """
    단일 파일 처리
    """
    try:
        # NIfTI 파일 로드
        nii = nib.load(file_path)
        data = nii.get_fdata()
        
        # 현재 라벨 값 확인
        unique_labels = np.unique(data)
        print(f"\n파일: {file_path}")
        print(f"  기존 라벨 값: {unique_labels}")
        
        # 라벨 리매핑
        remapped_data = remap_labels(data)
        
        # 리매핑 후 라벨 값 확인
        new_unique_labels = np.unique(remapped_data)
        print(f"  변환 후 라벨 값: {new_unique_labels}")
        
        # 새로운 NIfTI 이미지 생성 (원본과 동일한 affine, header 사용)
        new_nii = nib.Nifti1Image(remapped_data.astype(np.int16), nii.affine, nii.header)
        
        # 파일 저장 (원본 덮어쓰기)
        nib.save(new_nii, file_path)
        print(f"  ✓ 저장 완료")
        
        return True
    except Exception as e:
        print(f"\n에러 발생 ({file_path}): {str(e)}")
        return False


def main():
    # 기본 경로
    base_path = Path("/home/seoooa/project/coronary-artery/data/imageCAS_ablation")
    
    # train, valid, test 폴더 모두 처리
    subsets = ['train', 'valid', 'test']
    
    all_files = []
    
    # 모든 ventricle_atrium_combined.nii.gz 파일 찾기
    for subset in subsets:
        subset_path = base_path / subset
        if subset_path.exists():
            files = list(subset_path.glob("*/ventricle_atrium_combined.nii.gz"))
            all_files.extend(files)
            print(f"{subset}: {len(files)}개 파일 발견")
    
    print(f"\n총 {len(all_files)}개 파일을 처리합니다.\n")
    
    # 사용자 확인
    response = input("계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    # 각 파일 처리
    success_count = 0
    fail_count = 0
    
    print("\n처리 시작...\n")
    for file_path in tqdm(all_files, desc="파일 처리 중"):
        if process_file(file_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n\n처리 완료!")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")


if __name__ == "__main__":
    main()
