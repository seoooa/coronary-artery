"""Inference dataloader shared by baseline and no-coronary AGC comparisons."""

from __future__ import annotations

import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import (
    AsDiscreted,
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    GaussianSmoothd,
    LoadImaged,
    MapTransform,
    Orientationd,
    ScaleIntensityRanged,
)

from src.data.proposed_no_coronary_dataloader import convert_distance_map

from .anatomical_roi import build_anatomical_roi, voxel_spacing_from_affine


ANATOMY_FILENAME = "heart_combined_no_coronary.nii.gz"


class BuildAnatomicalROId(MapTransform):
    """Create a full-volume ROI before the common foreground crop."""

    def __init__(self, source_key: str, output_key: str, tau_mm: float):
        super().__init__([source_key])
        self.source_key = source_key
        self.output_key = output_key
        self.tau_mm = float(tau_mm)

    def __call__(self, data):
        result = dict(data)
        anatomy = result[self.source_key]
        affine = anatomy.affine
        spacing = voxel_spacing_from_affine(affine)
        roi = build_anatomical_roi(anatomy, spacing, self.tau_mm)
        result[self.output_key] = MetaTensor(
            torch.from_numpy(roi[None].astype(np.uint8)),
            affine=affine.clone() if isinstance(affine, torch.Tensor) else affine,
        )
        return result


class ConvertDistanceMapd(MapTransform):
    """Dictionary transform wrapping the established no-coronary SDM conversion."""

    def __init__(self, key: str = "seg"):
        super().__init__([key])
        self.key = key

    def __call__(self, data):
        result = dict(data)
        if self.key == "seg":
            return convert_distance_map(result)
        temporary = {"seg": result[self.key]}
        result[self.key] = convert_distance_map(temporary)["seg"]
        return result


class AnatomicalROIPostProcessingDataModule(pl.LightningDataModule):
    """Provide aligned CT, GT, ROI, and optional seven-channel conditioning."""

    def __init__(
        self,
        mode: str,
        tau_mm: float,
        data_dir: str = "data/imageCAS",
        anatomy_dir: str = "data/imageCAS_no_coronary_conditioning",
        guide: str = "distanceMap",
        num_workers: int = 4,
    ):
        super().__init__()
        if mode not in {"baseline", "proposed_no_coronary"}:
            raise ValueError(f"unsupported mode: {mode}")
        if guide not in {"segMap", "distanceMap"}:
            raise ValueError(f"unsupported guide: {guide}")
        self.mode = mode
        self.tau_mm = float(tau_mm)
        self.data_dir = Path(data_dir)
        self.anatomy_dir = Path(anatomy_dir)
        self.guide = guide
        self.num_workers = num_workers
        self.test_ds = None

    def load_data_split(self, split: str = "test"):
        split_dir = self.data_dir / split
        anatomy_split_dir = self.anatomy_dir / split
        if not split_dir.is_dir() or not anatomy_split_dir.is_dir():
            raise FileNotFoundError(f"missing data split: {split}")

        records = []
        missing = []
        for case in sorted(os.listdir(split_dir)):
            case_dir = split_dir / case
            if not case_dir.is_dir():
                continue
            paths = {
                "image": case_dir / "img.nii.gz",
                "label": case_dir / "label.nii.gz",
                "anatomy": anatomy_split_dir / case / ANATOMY_FILENAME,
            }
            absent = [str(path) for path in paths.values() if not path.is_file()]
            if absent:
                missing.append((case, absent))
            else:
                records.append({key: str(path) for key, path in paths.items()})

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} incomplete {split} cases; examples: {missing[:5]}"
            )
        if not records:
            raise RuntimeError(f"no complete cases found in {split}")
        return records

    def _transform(self):
        transforms = [
            LoadImaged(keys=["image", "label", "anatomy"]),
            EnsureChannelFirstd(keys=["image", "label", "anatomy"]),
            Orientationd(keys=["image", "label", "anatomy"], axcodes="RAS"),
            BuildAnatomicalROId("anatomy", "anatomical_roi", self.tau_mm),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-150,
                a_max=550,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]

        crop_keys = ["image", "label", "anatomy", "anatomical_roi"]
        if self.mode == "proposed_no_coronary":
            transforms.append(CopyItemsd(keys=["anatomy"], names=["seg"]))
            crop_keys.append("seg")

        transforms.append(CropForegroundd(keys=crop_keys, source_key="image"))

        if self.mode == "proposed_no_coronary":
            # One-hot encode after cropping to avoid materializing seven full-volume
            # channels while preserving exactly the same conditioning values.
            transforms.append(AsDiscreted(keys=["seg"], to_onehot=7))
            if self.guide == "distanceMap":
                transforms.append(ConvertDistanceMapd("seg"))
            else:
                transforms.append(GaussianSmoothd(keys=["seg"], sigma=1.0))
        return Compose(transforms)

    def setup(self, stage=None):
        records = self.load_data_split("test")
        self.test_ds = Dataset(data=records, transform=self._transform())
        print(f"Found {len(records)} test cases for {self.mode}")

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
