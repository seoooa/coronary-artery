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
    Lambda,
)
from monai.data import CacheDataset, DataLoader, Dataset
import os
import SimpleITK as sitk
import torch
import lightning.pytorch as pl
from pathlib import Path
import yaml
from src.data.transforms import AddSkeletonToDatad

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

def ConvertDistanceMap(data):
    """
    convert segmentation to distance map
    """
    seg = data["seg"]  # [C, H, W, D] one-hot encoded segmentation
    
    distance_map = create_distance_map(seg)
    
    data["seg"] = distance_map
    return data

class CoronaryArterySkelRECDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imageCAS",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.1,
        use_distance_map: bool = False,
        use_skeleton: bool = True,
        skeleton_do_tube: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.use_distance_map = use_distance_map
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.use_skeleton = use_skeleton
        self.skeleton_do_tube = skeleton_do_tube

    def load_data_splits(self, split: str):
        split_dir = self.data_dir / split
        cases = sorted(os.listdir(split_dir))
        
        data_files = []
        for case in cases:
            case_dir = split_dir / case

            image_file = str(case_dir / "img.nii.gz")
            label_file = str(case_dir / "label.nii.gz")
            seg_file = str(case_dir / "heart_combined.nii.gz")  # roi segmentation
            
            if os.path.exists(image_file) and os.path.exists(label_file):
                data_files.append({
                    "image": image_file,
                    "label": label_file,
                    "seg": seg_file
                })
        
        return data_files

    def prepare_data(self):
        # Base keys for transforms
        base_keys = ["image", "label", "seg"]
        
        # Training transforms
        transforms = [
            LoadImaged(keys=base_keys),
            EnsureChannelFirstd(keys=base_keys),
            Orientationd(keys=base_keys, axcodes="RAS"),
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
            CropForegroundd(keys=base_keys, source_key="image"),
        ]

        # Add skeleton generation if enabled
        if self.use_skeleton:
            transforms.append(
                AddSkeletonToDatad(
                    keys=["label"], 
                    do_tube=self.skeleton_do_tube,
                    allow_missing_keys=False
                )
            )
            # Update keys to include skeleton for subsequent transforms
            base_keys_with_skel = base_keys + ["skeleton"]
        else:
            base_keys_with_skel = base_keys

        # distance map 
        if self.use_distance_map:
            transforms.append(Lambda(ConvertDistanceMap))

        transforms.extend([
            RandCropByPosNegLabeld(
                keys=base_keys_with_skel,
                label_key="label",
                spatial_size=self.patch_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=base_keys_with_skel,
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=base_keys_with_skel,
                spatial_axis=[1],
                prob=0.10,
            ),
        ])

        if not self.use_distance_map:
            transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))

        transforms.append(RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5))

        self.train_transforms = Compose(transforms)

        # Validation transforms (no augmentation)
        val_transforms = [
            LoadImaged(keys=base_keys),
            EnsureChannelFirstd(keys=base_keys),
            Orientationd(keys=base_keys, axcodes="RAS"),
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
            CropForegroundd(keys=base_keys, source_key="image"),
        ]

        # Add skeleton generation for validation too if enabled
        if self.use_skeleton:
            val_transforms.append(
                AddSkeletonToDatad(
                    keys=["label"], 
                    do_tube=self.skeleton_do_tube,
                    allow_missing_keys=False
                )
            )

        # distance map
        if self.use_distance_map:
            val_transforms.append(Lambda(ConvertDistanceMap))

        if not self.use_distance_map:
            val_transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        # debug transforms
        if self.use_distance_map:
            print("DISTANCE MAP Guided Training")
        else:
            print("SEGMENTATION MAP Guided Training")
            
        if self.use_skeleton:
            print(f"SKELETON RECALL Training enabled (tube dilation: {self.skeleton_do_tube})")

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
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            # # prefetch_factor=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=False,
        )