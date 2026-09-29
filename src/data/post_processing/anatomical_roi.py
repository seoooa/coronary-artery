"""Shared anatomical ROI and connected-component post-processing operations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from monai.transforms import KeepLargestConnectedComponent
from scipy.ndimage import distance_transform_edt


def _as_spatial_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a 3-D NumPy array after removing singleton channel dimensions."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D spatial array, got shape {array.shape}")
    return array


def voxel_spacing_from_affine(affine: np.ndarray | torch.Tensor) -> tuple[float, ...]:
    """Extract positive voxel sizes in array-axis order from a 4x4 affine."""
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got shape {affine.shape}")
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"invalid voxel spacing derived from affine: {spacing}")
    return tuple(float(value) for value in spacing)


def compute_anatomical_distance(
    anatomy_label_map: np.ndarray | torch.Tensor,
    spacing_mm: Sequence[float],
) -> np.ndarray:
    """Compute the outside distance in millimetres to the six-structure union."""
    anatomy = _as_spatial_numpy(anatomy_label_map)
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError(f"invalid spacing_mm: {spacing_mm}")

    labels = set(np.unique(anatomy).tolist())
    unexpected = labels - set(range(7))
    if unexpected:
        raise ValueError(f"unexpected no-coronary anatomy labels: {sorted(unexpected)}")

    anatomy_union = anatomy > 0
    if not anatomy_union.any():
        raise ValueError("the six-structure anatomy union is empty")

    # distance_transform_edt is zero inside the union and positive outside it.
    return distance_transform_edt(~anatomy_union, sampling=spacing)


def build_anatomical_roi(
    anatomy_label_map: np.ndarray | torch.Tensor,
    spacing_mm: Sequence[float],
    tau_mm: float,
) -> np.ndarray:
    """Build R_tau = 1[D(x) <= tau] as a boolean 3-D mask."""
    if not np.isfinite(tau_mm) or tau_mm < 0:
        raise ValueError(f"tau_mm must be finite and non-negative, got {tau_mm}")
    return compute_anatomical_distance(anatomy_label_map, spacing_mm) <= tau_mm


def apply_anatomical_roi(
    prediction_onehot: torch.Tensor,
    roi_mask: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Intersect foreground with ROI and rebuild a valid two-channel one-hot map."""
    if prediction_onehot.ndim != 4 or prediction_onehot.shape[0] != 2:
        raise ValueError(
            "prediction_onehot must have shape [2, H, W, D], "
            f"got {tuple(prediction_onehot.shape)}"
        )

    roi = torch.as_tensor(
        _as_spatial_numpy(roi_mask), device=prediction_onehot.device, dtype=torch.bool
    )
    if tuple(roi.shape) != tuple(prediction_onehot.shape[1:]):
        raise ValueError(
            f"ROI shape {tuple(roi.shape)} does not match prediction shape "
            f"{tuple(prediction_onehot.shape[1:])}"
        )

    foreground = prediction_onehot[1].bool() & roi
    return torch.stack((~foreground, foreground)).to(prediction_onehot.dtype)


def keep_largest_components(
    prediction_onehot: torch.Tensor,
    num_components: int = 2,
) -> torch.Tensor:
    """Keep the requested number of largest foreground components."""
    if num_components < 1:
        raise ValueError("num_components must be at least 1")
    if prediction_onehot.ndim != 4 or prediction_onehot.shape[0] != 2:
        raise ValueError(
            "prediction_onehot must have shape [2, H, W, D], "
            f"got {tuple(prediction_onehot.shape)}"
        )
    transform = KeepLargestConnectedComponent(
        applied_labels=[1],
        is_onehot=True,
        independent=True,
        num_components=num_components,
    )
    processed = transform(prediction_onehot)
    foreground = processed[1].bool()
    return torch.stack((~foreground, foreground)).to(prediction_onehot.dtype)
