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

class CoronaryArteryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imageCAS",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.05,
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
        
    def load_data_splits(self, split: str):
        split_dir = self.data_dir / split
        cases = sorted(os.listdir(split_dir))
        
        data_files = []
        for case in cases:
            case_dir = split_dir / case

            image_file = str(case_dir / "img.nii.gz")
            label_file = str(case_dir / "label.nii.gz")
            pos_file = str(case_dir / "ppe.nii.gz")  # positional embedding
            
            if os.path.exists(image_file) and os.path.exists(label_file) and os.path.exists(pos_file):
                data_files.append({
                    "image": image_file,
                    "label": label_file,
                    "pos": pos_file
                })
        
        return data_files

    def prepare_data(self):
        transforms = [
            LoadImaged(keys=["image", "label", "pos"]),
            EnsureChannelFirstd(keys=["image", "label", "pos"]),
            Orientationd(keys=["image", "label", "pos"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "pos"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label", "pos"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label", "pos"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label", "pos"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5)
        ]

        self.train_transforms = Compose(transforms)

        val_transforms = [
            LoadImaged(keys=["image", "label", "pos"]),
            EnsureChannelFirstd(keys=["image", "label", "pos"]),
            Orientationd(keys=["image", "label", "pos"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "pos"], source_key="image"),
        ]

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        print("Positional Embedding (PPE) Training")

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