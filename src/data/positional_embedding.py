import autorootcwd
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from pathlib import Path
from src.utils.ape.model import APE
from tqdm import tqdm
import click

def load_image(image_path: str) -> tuple[torch.Tensor, nib.Nifti1Image]:
    """Load NIfTI image and preprocess it."""
    # Load NIfTI image
    nifti_img = nib.load(image_path)
    image = nifti_img.get_fdata()
    
    # Convert to float32 and normalize (0-1 range)
    image = np.float32(image)
    image = (image - image.min()) / (image.max() - image.min())
    
    # Add batch and channel dimensions [B, C, H, W, D]
    image = torch.from_numpy(image)[None, None]
    
    return image, nifti_img


def save_as_nifti(data: torch.Tensor, affine: np.ndarray, output_path: Path):
    """Save tensor as NIfTI image."""
    # Move to CPU and convert to numpy
    data_np = data.cpu().numpy()
    
    # Process batch and channel dimensions [B, C, H, W, D] -> [H, W, D, C]
    data_np = np.squeeze(data_np, axis=0)  # Remove batch dimension
    data_np = np.moveaxis(data_np, 0, -1)  # Move channel to last dimension
    
    # Create NIfTI image and save
    nifti_img = nib.Nifti1Image(data_np, affine)
    nib.save(nifti_img, output_path)

def save_as_numpy(data: torch.Tensor, output_path: Path):
    """Save tensor as NumPy (.npy) file."""
    # Move to CPU and convert to numpy
    data_np = data.cpu().numpy()
    
    # Save as NumPy file
    np.save(output_path, data_np)
    
def inference(
    model: APE,
    image: torch.Tensor,
    mask: torch.Tensor = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """Use APE model to calculate anatomical positional embeddings of images."""
    model = model.to(device)
    model.eval()
    
    image = image.to(device)
    if mask is not None:
        mask = mask.to(device)

    _, _, H, W, D = image.shape
    
    with torch.no_grad():
        # Calculate basic APE maps (shape: [B, 3, H, W, D])
        ape_maps = model(image, mask)

        ape_maps_upsampled = F.interpolate(
            ape_maps, 
            size=(H, W, D),  # original size
            mode='trilinear',  # trilinear interpolation
            align_corners=False  # generally recommended
        )
        
        # Convert to sin-cos embeddings (shape: [B, embed_dim, H, W, D])
        embeddings = model.to_sin_cos(ape_maps_upsampled)

    return ape_maps_upsampled, embeddings


def process_directory(input_dir: Path, output_dir: Path, model: APE, device: str, group_dirs=None):
    """Generate positional embeddings for all images in the given directory."""
    # Get list of patient directories to process
    if group_dirs is None:
        patient_dirs = [d for d in input_dir.glob("*") if d.is_dir()]
    else:
        patient_dirs = group_dirs
    
    # Create progress bar
    pbar = tqdm(patient_dirs, desc=f"Processing {input_dir.name}")
    
    # Process all patient directories
    for patient_dir in pbar:
        # Input image path
        img_path = patient_dir / "img.nii.gz"
        if not img_path.exists():
            pbar.write(f"Warning: {img_path} not found.")
            continue
            
        # Create output directory
        patient_output_dir = output_dir / patient_dir.name
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update progress bar description
        pbar.set_description(f"Processing {patient_dir.name}")
        
        # Load and process image
        image, nifti_img = load_image(str(img_path))
        
        # Run inference
        ape_maps, embeddings = inference(model, image, device=device)
        
        # Save embeddings
        save_as_numpy(ape_maps, patient_output_dir / "ape_maps.npy")
        # save_as_numpy(embeddings, patient_output_dir / "pos_embed.npy")

        # save_as_nifti(embeddings, nifti_img.affine, patient_output_dir / "pos_embed.nii.gz")

        pbar.write(f"Completed: {patient_output_dir}")


@click.command()
@click.option('--gpu_number', default=0, type=int, help='GPU device number to use')
def main(gpu_number):
    # Set GPU device
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model (automatically download pretrained weights from HuggingFace)
    print("Loading APE model...")
    model = APE(pretrained=True)
    print("Model loaded successfully!")
    
    # Set base paths
    base_input = Path("data/imageCAS_heart")
    base_output = Path("data/imageCAS_heart_pos")
    
    # Get available datasets
    datasets = [d.name for d in base_input.glob("*") if d.is_dir() and d.name in ["train", "valid", "test"]]
    
    # Process each dataset with overall progress
    for dataset in tqdm(datasets, desc="Datasets", position=0):
        input_dir = base_input / dataset
        output_dir = base_output / dataset
        
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist.")
            continue
        
        print(f"\n=== Processing {dataset} dataset ===")
        
        if dataset == 'train':
            # Get all train patient directories
            patient_dirs = sorted([d for d in input_dir.glob("*") if d.is_dir()])
            total_patients = len(patient_dirs)
            
            # Divide into three groups
            group1_size = total_patients // 3
            group2_size = (total_patients - group1_size) // 2
            group3_size = total_patients - group1_size - group2_size
            
            groups = [
                ("group1", patient_dirs[:group1_size]),
                ("group2", patient_dirs[group1_size:group1_size + group2_size]),
                ("group3", patient_dirs[group1_size + group2_size:])
            ]
            
            for group_name, group_dirs in groups:
                print(f"\n===== Processing Train {group_name} ({len(group_dirs)} patients) =====")
                process_directory(input_dir, output_dir, model, device, group_dirs)
                print(f"===== Train {group_name} completed =====")
        else:
            process_directory(input_dir, output_dir, model, device)
            
        print(f"=== {dataset} dataset processing completed ===\n")


if __name__ == "__main__":
    main()