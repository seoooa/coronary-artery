"""Validation-only calibration for the anatomical ROI distance threshold."""

from __future__ import annotations

import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from .anatomical_roi import compute_anatomical_distance, voxel_spacing_from_affine


ANATOMY_FILENAME = "heart_combined_no_coronary.nii.gz"
ANATOMY_LABELS = {
    "1": "aorta",
    "2": "heart_myocardium",
    "3": "heart_ventricle_left",
    "4": "heart_ventricle_right",
    "5": "heart_atrium_left",
    "6": "heart_atrium_right",
}


def _load_binary_gt(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"GT contains NaN or infinite values: {path}")
    return data > 0, image


def _load_anatomy(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"anatomy contains NaN or infinite values: {path}")
    integer_data = data.astype(np.uint8)
    if not np.array_equal(data, integer_data):
        raise ValueError(f"anatomy contains non-integer labels: {path}")
    return integer_data, image


def _coverage(distances: np.ndarray, tau_mm: float) -> float:
    return float(np.count_nonzero(distances <= tau_mm) / distances.size)


def calibrate_threshold(
    data_dir: str | Path = "data/imageCAS",
    anatomy_dir: str | Path = "data/imageCAS_no_coronary_conditioning",
    coverage_target: float = 0.995,
    round_up_mm: float = 0.5,
) -> dict[str, object]:
    """Calibrate one global tau from validation GT target-voxel distances."""
    if not 0 < coverage_target <= 1:
        raise ValueError("coverage_target must be in (0, 1]")
    if round_up_mm < 0:
        raise ValueError("round_up_mm must be non-negative")

    data_dir = Path(data_dir)
    anatomy_dir = Path(anatomy_dir)
    validation_dir = data_dir / "valid"
    anatomy_validation_dir = anatomy_dir / "valid"
    if not validation_dir.is_dir() or not anatomy_validation_dir.is_dir():
        raise FileNotFoundError("validation data or anatomy directory is missing")

    case_distances: list[tuple[str, np.ndarray]] = []
    for case_dir in sorted(path for path in validation_dir.iterdir() if path.is_dir()):
        gt_path = case_dir / "label.nii.gz"
        anatomy_path = anatomy_validation_dir / case_dir.name / ANATOMY_FILENAME
        if not gt_path.is_file() or not anatomy_path.is_file():
            raise FileNotFoundError(
                f"missing validation input for case {case_dir.name}: "
                f"GT={gt_path.is_file()}, anatomy={anatomy_path.is_file()}"
            )

        gt, gt_image = _load_binary_gt(gt_path)
        anatomy, anatomy_image = _load_anatomy(anatomy_path)
        if gt.shape != anatomy.shape:
            raise ValueError(
                f"shape mismatch for case {case_dir.name}: {gt.shape} vs {anatomy.shape}"
            )
        if not np.allclose(gt_image.affine, anatomy_image.affine):
            raise ValueError(f"affine mismatch for validation case {case_dir.name}")
        if not gt.any():
            raise ValueError(f"empty GT target for validation case {case_dir.name}")

        spacing = voxel_spacing_from_affine(anatomy_image.affine)
        distance_mm = compute_anatomical_distance(anatomy, spacing)
        case_distances.append((case_dir.name, distance_mm[gt].astype(np.float32)))

    if not case_distances:
        raise RuntimeError("no validation cases were found")

    pooled_distances = np.concatenate([distances for _, distances in case_distances])
    tau_unrounded = float(
        np.quantile(pooled_distances, coverage_target, method="higher")
    )
    tau_mm = tau_unrounded
    if round_up_mm > 0:
        tau_mm = math.ceil(tau_unrounded / round_up_mm) * round_up_mm

    subject_coverages = {
        case_id: _coverage(distances, tau_mm)
        for case_id, distances in case_distances
    }
    result: dict[str, object] = {
        "tau_mm": float(tau_mm),
        "tau_unrounded_mm": tau_unrounded,
        "coverage_target": float(coverage_target),
        "round_up_mm": float(round_up_mm),
        "validation_cases": len(case_distances),
        "validation_target_voxels": int(pooled_distances.size),
        "pooled_gt_coverage": _coverage(pooled_distances, tau_mm),
        "mean_subject_coverage": float(np.mean(list(subject_coverages.values()))),
        "minimum_subject_coverage": float(np.min(list(subject_coverages.values()))),
        "maximum_subject_coverage": float(np.max(list(subject_coverages.values()))),
        "anatomy_labels": ANATOMY_LABELS,
        "data_dir": str(data_dir),
        "anatomy_dir": str(anatomy_dir),
        "subject_coverages": subject_coverages,
    }
    return result


def load_or_calibrate_threshold(
    config_path: str | Path,
    data_dir: str | Path,
    anatomy_dir: str | Path,
    coverage_target: float = 0.995,
    round_up_mm: float = 0.5,
    recalibrate: bool = False,
) -> dict[str, object]:
    """Reuse a matching config or calibrate and save a new validation config."""
    config_path = Path(config_path)
    if config_path.is_file() and not recalibrate:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        if not np.isclose(config.get("coverage_target"), coverage_target):
            raise ValueError(
                "existing threshold config uses a different coverage_target; "
                "pass --recalibrate or select another result directory"
            )
        if not np.isclose(config.get("round_up_mm"), round_up_mm):
            raise ValueError(
                "existing threshold config uses a different round_up_mm; "
                "pass --recalibrate or select another result directory"
            )
        return config

    config = calibrate_threshold(
        data_dir=data_dir,
        anatomy_dir=anatomy_dir,
        coverage_target=coverage_target,
        round_up_mm=round_up_mm,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return config
