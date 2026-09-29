"""Data module for seven-channel conditioning without coronary arteries."""

from __future__ import annotations

import os
from pathlib import Path

import autorootcwd
import lightning.pytorch as pl
import SimpleITK as sitk
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    GaussianSmoothd,
    Lambda,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
)


NUM_SEG_CHANNELS = 7
SEG_FILENAME = "heart_combined_no_coronary.nii.gz"


def create_distance_map(binary_mask: torch.Tensor) -> torch.Tensor:
    """Create one signed distance map per conditioning channel."""
    distance_maps = []
    for channel in binary_mask:
        channel_mask = channel.numpy()
        sitk_mask = sitk.GetImageFromArray(channel_mask)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        sitk_mask.SetSpacing((0.35, 0.35, 0.5))
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_mask,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        )
        distance_maps.append(torch.from_numpy(sitk.GetArrayFromImage(distance_map)))
    return torch.stack(distance_maps)


def convert_distance_map(data):
    data["seg"] = create_distance_map(data["seg"])
    return data


class CoronaryArteryNoCoronaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imageCAS",
        conditioning_dir: str = "data/imageCAS_no_coronary_conditioning",
        batch_size: int = 4,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        num_workers: int = 4,
        cache_rate: float = 0.05,
        use_distance_map: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.conditioning_dir = Path(conditioning_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.use_distance_map = use_distance_map
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_data_splits(self, split: str):
        split_dir = self.data_dir / split
        conditioning_split_dir = self.conditioning_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"dataset split not found: {split_dir}")
        if not conditioning_split_dir.is_dir():
            raise FileNotFoundError(
                f"conditioning split not found: {conditioning_split_dir}"
            )

        data_files = []
        missing_cases = []
        for case in sorted(os.listdir(split_dir)):
            case_dir = split_dir / case
            if not case_dir.is_dir():
                continue

            files = {
                "image": case_dir / "img.nii.gz",
                "label": case_dir / "label.nii.gz",
                "seg": conditioning_split_dir / case / SEG_FILENAME,
            }
            missing = [str(path) for path in files.values() if not path.is_file()]
            if missing:
                missing_cases.append((case, missing))
                continue
            data_files.append({key: str(path) for key, path in files.items()})

        if missing_cases:
            examples = "; ".join(
                f"{case}: {paths}" for case, paths in missing_cases[:5]
            )
            raise FileNotFoundError(
                f"{split} has {len(missing_cases)} incomplete cases. Examples: {examples}"
            )
        if not data_files:
            raise RuntimeError(f"no complete cases found for split: {split}")
        return data_files

    def _base_transforms(self):
        return [
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
            AsDiscreted(keys=["seg"], to_onehot=NUM_SEG_CHANNELS),
            CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
        ]

    def prepare_data(self):
        transforms = self._base_transforms()
        if self.use_distance_map:
            transforms.append(Lambda(convert_distance_map))

        transforms.extend(
            [
                RandCropByPosNegLabeld(
                    keys=["image", "label", "seg"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label", "seg"], spatial_axis=[0], prob=0.10
                ),
                RandFlipd(
                    keys=["image", "label", "seg"], spatial_axis=[1], prob=0.10
                ),
            ]
        )
        if not self.use_distance_map:
            transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))
        transforms.append(RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5))
        self.train_transforms = Compose(transforms)

        val_transforms = self._base_transforms()
        if self.use_distance_map:
            val_transforms.append(Lambda(convert_distance_map))
        else:
            val_transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))
        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")
        print("No-coronary conditioning: 7 channels (background + 6 structures)")
        print(
            "DISTANCE MAP Guided Training"
            if self.use_distance_map
            else "SEGMENTATION MAP Guided Training"
        )

        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.val_transforms,
            cache_rate=0.0,
            num_workers=self.num_workers,
            copy_cache=False,
        )
        self.test_ds = CacheDataset(
            data=test_files,
            transform=self.val_transforms,
            cache_rate=0.0,
            num_workers=self.num_workers,
            copy_cache=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers)
