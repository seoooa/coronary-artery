import autorootcwd
import pandas as pd
import os
import shutil
from tqdm import tqdm

def organize_data_by_split(csv_path, source_dir, target_base_dir, split_column):
    df = pd.read_csv(csv_path, skiprows=1)
    
    df['FileName'] = df['FileName'].astype(str)

    subdirs = ['train', 'valid', 'test']
    for subdir in subdirs:
        target_dir = os.path.join(target_base_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
    
    # split data by Split-1
    train_cases = df[df[split_column] == 'Training']['FileName'].tolist()
    valid_cases = df[df[split_column] == 'Val']['FileName'].tolist()
    test_cases = df[df[split_column] == 'Testing']['FileName'].tolist()
    
    def copy_patient_data(case_id, target_subdir):
        source_path = os.path.join(source_dir, case_id)
        target_path = os.path.join(target_base_dir, target_subdir, case_id)
        
        if os.path.exists(source_path):
            try:
                shutil.copytree(source_path, target_path)
                return True
            except Exception as e:
                print(f"Error copying {case_id}: {str(e)}")
                return False
        else:
            print(f"Source directory not found for patient {case_id}")
            return False
    
    print("Copying training data...")
    for case in tqdm(train_cases):
        copy_patient_data(case, 'train')
    
    print("\nCopying validation data...")
    for case in tqdm(valid_cases):
        copy_patient_data(case, 'valid')
    
    print("\nCopying testing data...")
    for case in tqdm(test_cases):
        copy_patient_data(case, 'test')
    
    print("\n=== Data Organization Complete ===")
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(valid_cases)}")
    print(f"Testing cases: {len(test_cases)}")

if __name__ == "__main__":
    csv_path = "data/imageCAS/imageCAS_data_split.csv"
    source_dir = "data/imageCAS"
    target_base_dir = "data/imageCAS_split"

    organize_data_by_split(csv_path, source_dir, target_base_dir, split_column='Split-1')