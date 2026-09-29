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
import functools

# 7 experiments
EXPERIMENT_CONFIGS = {
    1: {
        "name": "Original Data (clipping x, normalization x)",
        "clip_min": None,
        "clip_max": None,
        "normalize": False,
        "norm_min": -1,
        "norm_max": 1
    },
    2: {
        "name": "Normalization (clipping x, normalization -1~1)",
        "clip_min": None,
        "clip_max": None,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    3: {
        "name": "Symmetric Clipping 1 (clipping -20~20, normalization -1~1)",
        "clip_min": -20.0,
        "clip_max": 20.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    4: {
        "name": "Symmetric Clipping 2 (clipping -50~50, normalization -1~1)",
        "clip_min": -50.0,
        "clip_max": 50.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    5: {
        "name": "Symmetric Clipping 3 (clipping -100~100, normalization -1~1)",
        "clip_min": -100.0,
        "clip_max": 100.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    6: {
        "name": "Negative Clipping (clipping -50~0, normalization -1~1)",
        "clip_min": -50.0,
        "clip_max": 0.0,
        "normalize": True,
        "norm_min": -1,
        "norm_max": 1
    },
    7: {
        "name": "Positive Clipping (clipping 0~150, normalization 0~1)",
        "clip_min": 0.0,
        "clip_max": 150.0,
        "normalize": True,
        "norm_min": 0,
        "norm_max": 1
    },
}


def clip_normalize_distance_map(data, clip_min=None, clip_max=None, normalize=False, norm_min=-1, norm_max=1):
    """
    clip and normalize distance map
    """
    dist = data["seg"]  # [8, H, W, D]
    
    # clipping and normalization
    distance_maps = []
    
    for c in range(dist.shape[0]):  # each channel

        channel_mask = dist[c].numpy()  # [H, W, D]
        
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
        distance_map_tensor = torch.from_numpy(distance_map_array)
        
        # clipping if clip_min and clip_max are provided
        if clip_min is not None and clip_max is not None:
            clipped_dist = torch.clamp(distance_map_tensor, min=clip_min, max=clip_max)
        else:
            clipped_dist = distance_map_tensor
        
        # normalize if normalize=True
        if normalize:
            if clip_min is not None and clip_max is not None:
                # use clipping range for normalization
                processed_dist = (clipped_dist - clip_min) / (clip_max - clip_min)
                processed_dist = processed_dist * (norm_max - norm_min) + norm_min
            else:
                # use actual data range for normalization
                data_min = clipped_dist.min()
                data_max = clipped_dist.max()
                processed_dist = (clipped_dist - data_min) / (data_max - data_min)
                processed_dist = processed_dist * (norm_max - norm_min) + norm_min
        else:
            processed_dist = clipped_dist
        
        distance_maps.append(processed_dist)
    
    processed_dist = torch.stack(distance_maps)

    
    data["seg"] = processed_dist
    return data

def clip_normalize_distance_map_with_config(data, experiment_id):
    """
    clip and normalize distance map with experiment config
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Invalid experiment_id: {experiment_id}. Must be 1-7.")
    
    exp_config = EXPERIMENT_CONFIGS[experiment_id]
    
    return clip_normalize_distance_map(
        data,
        clip_min=exp_config["clip_min"],
        clip_max=exp_config["clip_max"],
        normalize=exp_config["normalize"],
        norm_min=exp_config["norm_min"],
        norm_max=exp_config["norm_max"]
    )

class CoronaryArteryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imageCAS_ablation",
        batch_size: int = 4,
        patch_size: tuple = (96, 96, 96),
        num_workers: int = 4, 
        cache_rate: float = 0.05,
        experiment_id: int = 1
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.experiment_id = experiment_id
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        # setting experiment
        if experiment_id not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Invalid experiment_id: {experiment_id}. Must be 1-7.")
        
        self.exp_config = EXPERIMENT_CONFIGS[experiment_id]
        
    def get_distance_map_processor(self):
        """Create a picklable function for distance map processing"""
        exp_config = self.exp_config
        
        def process_distance_map(data):
            return clip_normalize_distance_map(
                data,
                clip_min=exp_config["clip_min"],
                clip_max=exp_config["clip_max"],
                normalize=exp_config["normalize"],
                norm_min=exp_config["norm_min"],
                norm_max=exp_config["norm_max"]
            )
        
        return process_distance_map
        
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
        # Use functools.partial to create a picklable function
        distance_map_processor = functools.partial(
            clip_normalize_distance_map_with_config,
            experiment_id=self.experiment_id
        )
        
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
            Lambda(distance_map_processor),
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
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5)
        ]

        self.train_transforms = Compose(transforms)

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
            Lambda(distance_map_processor)
        ]

        self.val_transforms = Compose(val_transforms)

    def setup(self, stage=None):
        train_files = self.load_data_splits("train")
        val_files = self.load_data_splits("valid")
        test_files = self.load_data_splits("test")

        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        print(f"Found {len(test_files)} test cases")

        # debug transforms
        print(f"Ablation 1: Experiment {self.experiment_id}: {self.exp_config['name']}")
        
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