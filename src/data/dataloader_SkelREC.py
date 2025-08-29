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
)
from monai.data import CacheDataset, DataLoader
import os
import lightning.pytorch as pl
from pathlib import Path
from src.data.transforms import AddSkeletonToDatad


class CoronaryArterySkelRECDataModule(pl.LightningDataModule):
    """
    Data module that supports skeleton recall loss training by generating skeleton data
    """

    def __init__(
        self,
        data_dir: str = "data/imageCAS",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4,
        cache_rate: float = 0.1,
        use_skeleton: bool = True,
        skeleton_do_tube: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
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
            
            if os.path.exists(image_file) and os.path.exists(label_file):
                data_files.append({
                    "image": image_file,
                    "label": label_file
                })
        
        return data_files

    def prepare_data(self):
        # Base transforms for all stages
        base_keys = ["image", "label"]
        
        # Training transforms
        train_transform_list = [
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
            CropForegroundd(keys=base_keys, source_key="image"),
        ]
        
        # Add skeleton generation if enabled
        if self.use_skeleton:
            train_transform_list.append(
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
        
        # Continue with augmentation transforms
        train_transform_list.extend([
            RandCropByPosNegLabeld(
                keys=base_keys_with_skel,
                label_key="label",
                spatial_size=self.patch_size,
                pos=1,
                neg=1,
                num_samples=6,
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
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5),
        ])
        
        self.train_transforms = Compose(train_transform_list)

        # Validation transforms (no augmentation)
        val_transform_list = [
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
            CropForegroundd(keys=base_keys, source_key="image"),
        ]
        
        # Add skeleton generation for validation too if enabled
        if self.use_skeleton:
            val_transform_list.append(
                AddSkeletonToDatad(
                    keys=["label"], 
                    do_tube=self.skeleton_do_tube,
                    allow_missing_keys=False
                )
            )
        
        self.val_transforms = Compose(val_transform_list)

    def setup(self, stage=None):
        # Load data splits
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers
        ) 