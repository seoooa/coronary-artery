"""
Signed Distance Map (SDM) Perturbation dataloader for robustness testing

Two types of perturbations:
1. Dilation/Erosion: Add constant offset to simulate over/under segmentation
2. Boundary Noise: Add Gaussian noise to simulate boundary uncertainty
"""
import autorootcwd
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    CropForegroundd,
    Compose,
    Spacingd,
    AsDiscreted,
    GaussianSmoothd,
    MapTransform,
)
from monai.data import CacheDataset, DataLoader, Dataset
import os
import SimpleITK as sitk
import torch
import lightning.pytorch as pl
from pathlib import Path
import numpy as np


def create_distance_map(binary_mask):
    """
    create distance map from binary mask
    
    Args:
        binary_mask (torch.Tensor): [C, H, W, D] binary mask
        
    Returns:
        torch.Tensor: [C, H, W, D] distance map
    """
    distance_maps = []
    
    for c in range(binary_mask.shape[0]):  # each channel
        channel_mask = binary_mask[c].numpy()  # [H, W, D]
        
        # convert to SimpleITK image
        sitk_mask = sitk.GetImageFromArray(channel_mask)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        
        # create distance map
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_mask,
            insideIsPositive=False,  # heart outside is positive
            squaredDistance=False,
            useImageSpacing=True     # physical distance (mm)
        )
        
        # convert to tensor
        distance_map_array = sitk.GetArrayFromImage(distance_map)
        distance_maps.append(torch.from_numpy(distance_map_array))
    
    return torch.stack(distance_maps)


class ConvertDistanceMapd(MapTransform):
    """
    Convert segmentation to distance map (Transform class)
    """
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]  # [C, H, W, D] one-hot encoded segmentation
            distance_map = create_distance_map(seg)
            d[key] = distance_map
        return d


class AddDilationErosiond(MapTransform):
    """
    Add random constant offset to each channel (Dilation/Erosion)
    
    Simulates over-segmentation (positive offset) or under-segmentation (negative offset)
    """
    def __init__(self, keys, offset_range=2.0, seed=None):
        super().__init__(keys)
        self.offset_range = offset_range  # Random offset range: [-offset_range, +offset_range]
        self.seed = seed
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d = self._add_offset(d, key)
        return d
    
    def _add_offset(self, data, key):
        """
        Add random offset to each channel
        
        Each channel gets EXACTLY -offset_range or +offset_range (not in between)
        Positive offset → Dilation (over-segmentation)
        Negative offset → Erosion (under-segmentation)
        
        NOTE: seed is NOT used here to ensure different offsets for each patient
        """
        sdm = data[key]  # [C, H, W, D] distance map
        
        # Convert to torch tensor if needed
        if isinstance(sdm, np.ndarray):
            sdm = torch.from_numpy(sdm)
        
        # Calculate statistics
        sdm_std = float(sdm.std())
        sdm_mean = float(sdm.mean())
        sdm_min = float(sdm.min())
        sdm_max = float(sdm.max())
        
        # Generate random offset for each channel: EXACTLY -offset_range or +offset_range
        num_channels = sdm.shape[0]
        # Generate 0 or 1 randomly, then convert to -1 or +1, then multiply by offset_range
        random_signs = torch.randint(0, 2, (num_channels,)) * 2 - 1  # -1 or +1
        offsets = random_signs.float() * self.offset_range  # -offset_range or +offset_range
        
        # Apply offset to each channel
        perturbed_sdm = sdm.clone()
        for c in range(num_channels):
            perturbed_sdm[c] = sdm[c] + offsets[c]
        
        data[key] = perturbed_sdm
        data["perturbation_type"] = "dilation_erosion"
        data["offset_range"] = self.offset_range
        data["channel_offsets"] = offsets.cpu().numpy()
        data["original_sdm_std"] = sdm_std
        data["original_sdm_mean"] = sdm_mean
        
        return data


class AddBoundaryNoised(MapTransform):
    """
    Add Gaussian noise to distance map (Boundary Uncertainty)
    
    Simulates uncertainty in boundary localization
    """
    def __init__(self, keys, noise_std=0.5, seed=None):
        super().__init__(keys)
        self.noise_std = noise_std
        self.seed = seed
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d = self._add_noise(d, key)
        return d
    
    def _add_noise(self, data, key):
        """
        Add Gaussian noise to distance map
        
        NOTE: seed is NOT used here to ensure different noise for each patient
        """
        sdm = data[key]  # [C, H, W, D] distance map
        
        # Convert to torch tensor if needed
        if isinstance(sdm, np.ndarray):
            sdm = torch.from_numpy(sdm)
        
        # Calculate statistics
        sdm_std = float(sdm.std())
        sdm_mean = float(sdm.mean())
        sdm_min = float(sdm.min())
        sdm_max = float(sdm.max())
        
        # Generate Gaussian noise (different for each patient)
        noise = torch.randn_like(sdm, dtype=sdm.dtype) * self.noise_std
        
        # Add noise to distance map
        perturbed_sdm = sdm + noise
        
        data[key] = perturbed_sdm
        data["perturbation_type"] = "boundary_noise"
        data["noise_std"] = self.noise_std
        data["original_sdm_std"] = sdm_std
        data["original_sdm_mean"] = sdm_mean
        
        # Calculate SNR
        if sdm_std > 0 and self.noise_std > 0:
            snr = sdm_std / self.noise_std
        else:
            snr = float('inf')
        
        data["snr"] = snr
        
        return data


class CoronaryArteryDataModuleSDMPerturbation(pl.LightningDataModule):
    """
    DataModule with SDM perturbation for robustness testing
    
    Two modes:
    - 'dilation_erosion': Add constant offset (simulates over/under segmentation)
    - 'boundary_noise': Add Gaussian noise (simulates boundary uncertainty)
    """
    def __init__(
        self,
        data_dir: str = "data/imageCAS",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.05,
        perturbation_mode: str = "boundary_noise",  # 'dilation_erosion' or 'boundary_noise'
        offset_range: float = 2.0,  # For dilation/erosion mode
        noise_std: float = 0.5,  # For boundary noise mode
        perturbation_only_test: bool = True,
        seed: int = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.perturbation_mode = perturbation_mode
        self.offset_range = offset_range
        self.noise_std = noise_std
        self.perturbation_only_test = perturbation_only_test
        self.seed = seed
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
    def load_data_splits(self, split: str):
        split_dir = self.data_dir / split
        cases = sorted(os.listdir(split_dir))
        
        data_files = []
        for case in cases:
            case_dir = split_dir / case

            image_file = str(case_dir / "img.nii.gz")
            label_file = str(case_dir / "label.nii.gz")
            seg_file = str(case_dir / "heart_combined.nii.gz")
            
            if os.path.exists(image_file) and os.path.exists(label_file):
                data_files.append({
                    "image": image_file,
                    "label": label_file,
                    "seg": seg_file
                })
        
        return data_files

    def prepare_data(self):
        # Training transforms
        transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(
                keys=["seg"],
                to_onehot=8,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        ]

        # Convert to distance map FIRST
        transforms.append(ConvertDistanceMapd(keys=["seg"]))

        # Add perturbation to training if not perturbation_only_test
        if not self.perturbation_only_test:
            if self.perturbation_mode == "dilation_erosion":
                transforms.append(AddDilationErosiond(keys=["seg"], offset_range=self.offset_range, seed=self.seed))
            elif self.perturbation_mode == "boundary_noise":
                transforms.append(AddBoundaryNoised(keys=["seg"], noise_std=self.noise_std, seed=self.seed))

        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label", "seg"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label", "seg"],
                spatial_axis=[1],
                prob=0.10,
            ),
        ])

        transforms.append(RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5))

        self.train_transforms = Compose(transforms)

        # Validation/Test transforms
        val_transforms = [
            LoadImaged(keys=["image", "label", "seg"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AsDiscreted(
                keys=["seg"],
                to_onehot=8,
            ),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        ]

        # Convert to distance map FIRST
        val_transforms.append(ConvertDistanceMapd(keys=["seg"]))

        # ALWAYS add perturbation to validation/test data
        if self.perturbation_mode == "dilation_erosion":
            val_transforms.append(AddDilationErosiond(keys=["seg"], offset_range=self.offset_range, seed=self.seed))
        elif self.perturbation_mode == "boundary_noise":
            val_transforms.append(AddBoundaryNoised(keys=["seg"], noise_std=self.noise_std, seed=self.seed))

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        print(f"Perturbation Mode: {self.perturbation_mode}")
        if self.perturbation_mode == "dilation_erosion":
            print(f"Offset Range: ±{self.offset_range} mm")
        elif self.perturbation_mode == "boundary_noise":
            print(f"Noise Std: {self.noise_std} mm")
        print(f"Perturbation Only Test: {self.perturbation_only_test}")

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
        )
