import autorootcwd
from totalsegmentator.python_api import totalsegmentator
import os
import nibabel as nib
import numpy as np
from pathlib import Path
import click
import shutil
from tqdm import tqdm

# Define the global label map (continuous labeling for all segmentations)
GLOBAL_LABEL_MAP = {
    # Heart Chambers (1-7)
    "heart/coronary_arteries.nii.gz": 1,
    "heart/aorta.nii.gz": 2,
    "heart/heart_myocardium.nii.gz": 3,
    "heart/heart_ventricle_left.nii.gz": 4,
    "heart/heart_ventricle_right.nii.gz": 5,
    "heart/heart_atrium_left.nii.gz": 6,
    "heart/heart_atrium_right.nii.gz": 7,
    # "heart/pulmonary_artery.nii.gz": 8,
    
    # Lung (8-9)
    "lung/lung.nii.gz": 9,
    "lung/lung_vessels.nii.gz": 10,
    
    # Artery (10-11)
    "artery/superior_vena_cava.nii.gz": 11,
    "artery/inferior_vena_cava.nii.gz": 12,
    
    # Skeleton (12)
    "skeleton/sternum.nii.gz": 13,
}

def filter_results(output_path, keep_rois):
    all_files = os.listdir(output_path)

    for file in all_files:
        if not any(roi in file for roi in keep_rois):
            os.remove(os.path.join(output_path, file))

def combine_segmentations(seg_folder, output_file, label_map, is_flat=False):
    """
    Combine segmentations based on the label map.
    
    Args:
        seg_folder: folder where segmentation files are located
        output_file: path to save the merged result
        label_map: mapping of file names to label values
        is_flat: True if files are directly under seg_folder, False if considering subfolders
    """

    ref_img = None
    if is_flat:
        # flat structure: all files are directly under seg_folder
        for rel_path, label_value in label_map.items():
            file_name = os.path.basename(rel_path)
            file_path = os.path.join(seg_folder, file_name)
            if os.path.exists(file_path):
                ref_img = nib.load(file_path)
                break
    else:
        # hierarchical structure: files are分散在子文件夹中
        for rel_path, label_value in label_map.items():
            file_path = os.path.join(seg_folder, rel_path)
            if os.path.exists(file_path):
                ref_img = nib.load(file_path)
                break
    
    if ref_img is None:
        print("[Error] Segmentation file not found.")
        return
    
    # get size and affine information from reference image
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    # initialize array to store merged segmentation (background is 0)
    combined_seg = np.zeros(ref_shape, dtype=np.uint8)
    
    # merge each segmentation file according to the label map
    for rel_path, label_value in label_map.items():
        if is_flat:
            file_name = os.path.basename(rel_path)
            file_path = os.path.join(seg_folder, file_name)
        else:
            file_path = os.path.join(seg_folder, rel_path)
            
        if os.path.exists(file_path):
            print(f"Merging: {rel_path} (label value: {label_value})")
            seg_img = nib.load(file_path)
            seg_data = seg_img.get_fdata().astype(np.uint8)
            
            # if there is already an assigned area, overwrite it according to priority
            # here, assume that higher label value has higher priority
            combined_seg = np.where(
                (seg_data > 0) & ((combined_seg == 0) | (label_value > combined_seg)),
                label_value,
                combined_seg
            )
    
    # save the final merged segmentation
    combined_img = nib.Nifti1Image(combined_seg, ref_affine)
    nib.save(combined_img, output_file)
    print(f"Saved merged segmentation: {output_file}")
    
    return output_file

def run_heart_segmentation(input_path, output_folder):
    """
    Run segmentation for heart-related areas.
    """
    # temporary output folder
    temp_folder = os.path.join(output_folder, "temp_heart")
    os.makedirs(temp_folder, exist_ok=True)
    
    # define the label map for heart-related areas
    label_map = {
        "coronary_arteries.nii.gz": 1,
        "aorta.nii.gz": 2,
        "heart_myocardium.nii.gz": 3,
        "heart_ventricle_left.nii.gz": 4,
        "heart_ventricle_right.nii.gz": 5,
        "heart_atrium_left.nii.gz": 6,
        "heart_atrium_right.nii.gz": 7,
        # "pulmonary_artery.nii.gz": 8,
    }
    
    try:
        # run heart chambers segmentation with heartchambers_highres
        print("Heart Chambers Segmentation Running...")
        totalsegmentator(
            input_path,
            temp_folder,
            task='coronary_arteries',
            device='gpu'
        )
        totalsegmentator(
            input_path,
            temp_folder,
            task='heartchambers_highres',
            device='gpu'
        )
        
        # check the file list in the temporary folder (for debugging)
        print("\nFile list in the temporary folder:")
        temp_files = os.listdir(temp_folder)
        for file in temp_files:
            print(f" - {file}")
        
        # create the merged segmentation
        combine_path = os.path.join(output_folder, "heart_combined.nii.gz")
        combine_segmentations(temp_folder, combine_path, label_map, is_flat=True)
        
        print("Heart segmentation completed!")
        return True
        
    except Exception as e:
        print(f"Heart segmentation failed: {str(e)}")
        return False
    finally:
        # delete the temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def run_lung_segmentation(input_path, output_folder):
    """
    Run segmentation for lung-related areas.
    """
    # temporary output folder
    temp_folder = os.path.join(output_folder, "temp_lung")
    os.makedirs(temp_folder, exist_ok=True)
    
    # define the label map for lung-related areas
    label_map = {
        "lung.nii.gz": 9,
        "lung_vessels.nii.gz": 10,
    }
    
    try:
        # run lung segmentation with lung_vessels task
        print("Lung segmentation running...")
        totalsegmentator(
            input_path,
            temp_folder,
            task='lung_vessels',
            device='gpu'
        )
        
        totalsegmentator(
            input_path,
            temp_folder,
            task='lung_nodules',
            device='gpu'
        )
        
        # create the merged segmentation
        combine_path = os.path.join(output_folder, "lung_combined.nii.gz")
        combine_segmentations(temp_folder, combine_path, label_map, is_flat=True)
        
        print("Lung segmentation completed!")
        return True
        
    except Exception as e:
        print(f"Lung segmentation failed: {str(e)}")
        return False
    finally:
        # delete the temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def run_artery_segmentation(input_path, output_folder):
    """
    Run segmentation for artery-related areas.
    """
    # temporary output folder
    temp_folder = os.path.join(output_folder, "temp_artery")
    os.makedirs(temp_folder, exist_ok=True)
    
    # define the label map for artery-related areas
    label_map = {
        "superior_vena_cava.nii.gz": 11,
        "inferior_vena_cava.nii.gz": 12,
    }
    
    try:
        # run artery segmentation with total task
        print("Artery segmentation running...")
        totalsegmentator(
            input_path,
            temp_folder,
            task='total',
            roi_subset=['superior_vena_cava', 'inferior_vena_cava'],
            device='gpu'
        )
        
        # create the merged segmentation
        combine_path = os.path.join(output_folder, "artery_combined.nii.gz")
        combine_segmentations(temp_folder, combine_path, label_map, is_flat=True)
        
        print("Artery segmentation completed!")
        return True
        
    except Exception as e:
        print(f"Artery segmentation failed: {str(e)}")
        return False
    finally:
        # delete the temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def run_skeleton_segmentation(input_path, output_folder):
    """
    Run segmentation for skeleton-related areas.
    """
    # temporary output folder
    temp_folder = os.path.join(output_folder, "temp_skeleton")
    os.makedirs(temp_folder, exist_ok=True)
    
    # define the label map for skeleton-related areas
    label_map = {
        "sternum.nii.gz": 13,
    }
    
    try:
        # run skeleton segmentation with total task
        print("Skeleton segmentation running...")
        totalsegmentator(
            input_path,
            temp_folder,
            task='total',
            roi_subset=['sternum'],
            device='gpu'
        )
        
        # create the merged segmentation
        combine_path = os.path.join(output_folder, "skeleton_combined.nii.gz")
        combine_segmentations(temp_folder, combine_path, label_map, is_flat=True)
        
        print("Skeleton segmentation completed!")
        return True
        
    except Exception as e:
        print(f"Skeleton segmentation failed: {str(e)}")
        return False
    finally:
        # delete the temporary folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def create_all_combined_segmentation(output_folder, selected_rois):
    """
    Merge all selected ROIs into a single file.
    """
    combined_files = {}
    combined_labels = {}
    
    # check the combined file path and label for each ROI
    if "heart" in selected_rois:
        heart_file = os.path.join(output_folder, "heart_combined.nii.gz")
        if os.path.exists(heart_file):
            combined_files["heart"] = heart_file
            combined_labels["heart"] = list(range(1, 8))  # 1-7 labels
    
    if "lung" in selected_rois:
        lung_file = os.path.join(output_folder, "lung_combined.nii.gz")
        if os.path.exists(lung_file):
            combined_files["lung"] = lung_file
            combined_labels["lung"] = [8, 9]  # 8-9 labels
    
    if "artery" in selected_rois:
        artery_file = os.path.join(output_folder, "artery_combined.nii.gz")
        if os.path.exists(artery_file):
            combined_files["artery"] = artery_file
            combined_labels["artery"] = [10, 11]  # 10-11 labels
    
    if "skeleton" in selected_rois:
        skeleton_file = os.path.join(output_folder, "skeleton_combined.nii.gz")
        if os.path.exists(skeleton_file):
            combined_files["skeleton"] = skeleton_file
            combined_labels["skeleton"] = [12]  # 12 label
    
    if not combined_files:
        print("No files to merge.")
        return None
    
    # load the representative file
    first_key = list(combined_files.keys())[0]
    ref_img = nib.load(combined_files[first_key])
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    # initialize the array to store merged segmentation
    combined_seg = np.zeros(ref_shape, dtype=np.uint8)
    
    # get the label values from each combined file
    for roi_name, file_path in combined_files.items():
        print(f"Merging: {roi_name} ({file_path})")
        seg_img = nib.load(file_path)
        seg_data = seg_img.get_fdata().astype(np.uint8)
        
        # copy the values that match the label range of the corresponding ROI
        for label in combined_labels[roi_name]:
            mask = (seg_data == label)
            combined_seg[mask] = label
    
    # save the final merged segmentation
    combined_filename = "_".join(selected_rois) + "_combined.nii.gz"
    combined_path = os.path.join(output_folder, combined_filename)
    combined_img = nib.Nifti1Image(combined_seg, ref_affine)
    nib.save(combined_img, combined_path)
    
    print(f"Integrated segmentation saved: {combined_path}")
    return combined_path

@click.command()
@click.option('--input_root', required=False, default="data/imageCAS_RAS_affine", help='Root directory containing train/valid/test folders')
@click.option('--output_root', required=False, default="data/imageCAS_heart", help='Root directory for output')
@click.option('--heart', is_flag=True, help='Run heart segmentation')
@click.option('--lung', is_flag=True, help='Run lung segmentation')
@click.option('--artery', is_flag=True, help='Run artery segmentation')
@click.option('--skeleton', is_flag=True, help='Run skeleton segmentation')
@click.option('--all', is_flag=True, help='Run all segmentations')
@click.option('--gpu_number', default=None, type=int, help='Number of GPUs to use. If not specified, all available GPUs will be used.')
def main(input_root, output_root, heart=False, lung=False, artery=False, skeleton=False, all=False, gpu_number=None):
    """
    Run segmentation for specific ROIs using TotalSegmentator and create merged segmentation files based on the label map.
    Processes all patient data in train, valid, test directories.
    """
    
    print(f"\n===== ROI Segmentation Tool =====")
    print(f"Input root directory: {input_root}")
    print(f"Output root directory: {output_root}")
    print(f"Using GPU: {gpu_number}")
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    
    # Create output root directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    
    # if all options are False and all option is also False, print help
    if not any([heart, lung, artery, skeleton, all]):
        print("\nError: At least one ROI option must be selected.")
        print("Usage: python roi_segmentation.py --help")
        return
    
    # run all segmentations
    if all:
        heart = lung = artery = skeleton = True
    
    # Prepare selected ROIs list and their label information
    selected_rois = []
    label_info = {}
    
    if heart:
        selected_rois.append("heart")
        label_info["heart"] = {
            "coronary_arteries": 1,
            "aorta": 2,
            "heart_myocardium": 3,
            "heart_ventricle_left": 4,
            "heart_ventricle_right": 5,
            "heart_atrium_left": 6,
            "heart_atrium_right": 7
        }
    
    if lung:
        selected_rois.append("lung")
        label_info["lung"] = {
            "lung": 9,
            "lung_vessels": 10
        }
    
    if artery:
        selected_rois.append("artery")
        label_info["artery"] = {
            "superior_vena_cava": 11,
            "inferior_vena_cava": 12
        }
    
    if skeleton:
        selected_rois.append("skeleton")
        label_info["skeleton"] = {
            "sternum": 13
        }
    
    # Save class information to output_root directory
    class_info_path = os.path.join(output_root, "combined_class_info.txt")
    with open(class_info_path, "w") as f:
        f.write("Combined Segmentation Class Information\n")
        f.write("=====================================\n\n")
        for roi_name, labels in label_info.items():
            f.write(f"{roi_name.upper()}\n")
            f.write("-" * len(roi_name) + "\n")
            for part_name, label_value in labels.items():
                f.write(f"Label {label_value}: {part_name}\n")
            f.write("\n")
    
    print(f"Class information saved to: {class_info_path}")
    
    # Process each dataset split (train, valid, test)
    for split in ['test']:
        split_input_dir = os.path.join(input_root, split)
        split_output_dir = os.path.join(output_root, split)
        
        if not os.path.exists(split_input_dir):
            print(f"\nSkipping {split} - directory not found: {split_input_dir}")
            continue
            
        print(f"\n===== Processing {split.upper()} dataset =====")
        
        # Get list of patient directories
        patient_dirs = sorted(os.listdir(split_input_dir))
        
        if split == 'train':
            # Divide the data into 3 groups for train
            groups = [
                # ("group1", patient_dirs[:200]),
                # ("group2", patient_dirs[200:500]),
                # ("group3", patient_dirs[500:700])
            ]
            
            for group_name, group_dirs in groups:
                print(f"\n===== Processing Train {group_name} ({len(group_dirs)} patients) =====")
                
                for patient_dir in tqdm(group_dirs, desc=f"Processing train {group_name}"):
                    process_patient(patient_dir, split_input_dir, split_output_dir, 
                                 selected_rois, label_info, results=[], completed_rois=[])
        else:
            # valid and test are processed at once
            for patient_dir in tqdm(patient_dirs, desc=f"Processing {split} patients"):
                process_patient(patient_dir, split_input_dir, split_output_dir,
                             selected_rois, label_info, results=[], completed_rois=[])

def process_patient(patient_dir, split_input_dir, split_output_dir, selected_rois, label_info, results, completed_rois):
    """Process a single patient's data"""
    patient_input_path = os.path.join(split_input_dir, patient_dir, "img.nii.gz")
    patient_output_dir = os.path.join(split_output_dir, patient_dir)
    
    if not os.path.exists(patient_input_path):
        tqdm.write(f"\nSkipping patient {patient_dir} - img.nii.gz not found")
        return
            
    tqdm.write(f"\n----- Processing patient: {patient_dir} -----")
    tqdm.write(f"Input image: {patient_input_path}")
    tqdm.write(f"Output directory: {patient_output_dir}")
    
    # create the output directory
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # run segmentation based on the selected ROI
    if "heart" in selected_rois:
        tqdm.write("\n----- Heart Segmentation Started -----")
        try:
            result = run_heart_segmentation(patient_input_path, patient_output_dir)
            results.append(("Heart Chambers", result))
            if result:
                completed_rois.append("heart")
        except Exception as e:
            tqdm.write(f"Error in heart chambers segmentation: {str(e)}")
            results.append(("Heart Chambers", False))
    
    if "lung" in selected_rois:
        tqdm.write("\n----- Lung Segmentation Started -----")
        try:
            result = run_lung_segmentation(patient_input_path, patient_output_dir)
            results.append(("Lung", result))
            if result:
                completed_rois.append("lung")
        except Exception as e:
            tqdm.write(f"Error in lung segmentation: {str(e)}")
            results.append(("Lung", False))
    
    if "artery" in selected_rois:
        tqdm.write("\n----- Artery Segmentation Started -----")
        try:
            result = run_artery_segmentation(patient_input_path, patient_output_dir)
            results.append(("Artery", result))
            if result:
                completed_rois.append("artery")
        except Exception as e:
            tqdm.write(f"Error in artery segmentation: {str(e)}")
            results.append(("Artery", False))
    
    if "skeleton" in selected_rois:
        tqdm.write("\n----- Skeleton Segmentation Started -----")
        try:
            result = run_skeleton_segmentation(patient_input_path, patient_output_dir)
            results.append(("Skeleton", result))
            if result:
                completed_rois.append("skeleton")
        except Exception as e:
            tqdm.write(f"Error in skeleton segmentation: {str(e)}")
            results.append(("Skeleton", False))
    
    # if multiple ROIs are selected, create a combined file
    if len(completed_rois) > 1:
        create_all_combined_segmentation(patient_output_dir, completed_rois)
    
    # summarize the results for this patient
    tqdm.write(f"\n===== Results for patient {patient_dir} =====")
    for roi, success in results:
        status = "Success" if success else "Failed"
        tqdm.write(f"{roi}: {status}")
    
    if len(completed_rois) > 1:
        tqdm.write(f"\nAll selected ROIs ({', '.join(completed_rois)}) have been merged into a single file.")
    
    tqdm.write(f"\nResults saved in: {patient_output_dir}")

if __name__ == "__main__":
    main()