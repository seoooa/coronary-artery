"""Compare Top-K connected components and anatomical ROI filtering.

One invocation calibrates the validation-only ROI threshold (or reuses a saved
configuration), then evaluates both the baseline and no-coronary AGC models.
By default only per-case CSV files are written. Pass ``--save_outputs`` to save
the foreground prediction NIfTI for every case and method.
"""

from __future__ import annotations

import csv
from pathlib import Path

import autorootcwd
import click
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from script.proposed_no_coronary_train import (
    CoronaryArteryNoCoronarySegmentModel,
)
from script.train import CoronaryArterySegmentModel as BaselineSegmentModel
from src.data.post_processing import (
    AnatomicalROIPostProcessingDataModule,
    apply_anatomical_roi,
    keep_largest_components,
    load_or_calibrate_threshold,
)
from src.metrics.metrics import MetricFactory


METRIC_KEYS = (
    "dice",
    "hausdorff",
    "iou",
    "precision",
    "recall",
    "cldice",
    "betti_0",
    "betti_1",
)

CSV_FIELDS = [
    "dice_score",
    "hausdorff_score",
    "iou_score",
    "precision_score",
    "recall_score",
    "cldice_score",
    "betti_0_score",
    "betti_1_score",
    "roi_gt_coverage",
    "removed_prediction_fraction",
    "removed_true_positive_voxels",
    "removed_false_positive_voxels",
    "patient_id",
]


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(np.asarray(value).item())


def _one_hot_from_foreground(foreground: torch.Tensor) -> torch.Tensor:
    foreground = foreground.bool()
    return torch.stack((~foreground, foreground)).float()


def _case_metrics(metrics, prediction, label) -> dict[str, float]:
    MetricFactory.calculate_metrics(metrics, [prediction], [label])
    aggregated = MetricFactory.aggregate_metrics(metrics)
    MetricFactory.reset_metrics(metrics)
    return {key: _as_float(aggregated[key]) for key in METRIC_KEYS}


def _removal_statistics(
    raw_prediction: torch.Tensor,
    processed_prediction: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, float | int]:
    raw = raw_prediction[1].bool()
    processed = processed_prediction[1].bool()
    target = label[1].bool()
    removed = raw & ~processed
    raw_count = int(raw.sum().item())
    return {
        "removed_prediction_fraction": (
            float(removed.sum().item() / raw_count) if raw_count else 0.0
        ),
        "removed_true_positive_voxels": int((removed & target).sum().item()),
        "removed_false_positive_voxels": int((removed & ~target).sum().item()),
    }


def _save_output(
    output_path: Path,
    prediction: torch.Tensor,
    affine: torch.Tensor | np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    foreground = prediction[1].detach().cpu().numpy().astype(np.uint8)
    nib.save(nib.Nifti1Image(foreground, np.asarray(affine)), output_path)


def _write_result_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

        summary: dict[str, object] = {field: "" for field in CSV_FIELDS}
        for key in CSV_FIELDS:
            if not key.endswith("_score"):
                continue
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            summary[key] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        summary["roi_gt_coverage"] = (
            f"{np.mean([float(row['roi_gt_coverage']) for row in rows]):.6f}"
            if rows and rows[0]["roi_gt_coverage"] != ""
            else ""
        )
        summary["removed_prediction_fraction"] = (
            f"{np.mean([float(row['removed_prediction_fraction']) for row in rows]):.6f}"
        )
        summary["removed_true_positive_voxels"] = int(
            sum(int(row["removed_true_positive_voxels"]) for row in rows)
        )
        summary["removed_false_positive_voxels"] = int(
            sum(int(row["removed_false_positive_voxels"]) for row in rows)
        )
        summary["patient_id"] = "AVG ± STD"
        writer.writerow(summary)
    temporary_path.replace(path)


def _read_result_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"unexpected CSV columns in existing result: {path}")
        return [row for row in reader if row["patient_id"] != "AVG ± STD"]


def _patient_id(batch) -> str:
    filename = batch["image"].meta["filename_or_obj"][0]
    return Path(str(filename)).parent.name


def _cropped_affine(batch) -> np.ndarray:
    affine = batch["image"].affine
    if isinstance(affine, torch.Tensor) and affine.ndim == 3:
        affine = affine[0]
    return np.asarray(affine.detach().cpu() if isinstance(affine, torch.Tensor) else affine)


def _raw_prediction(logits: torch.Tensor) -> torch.Tensor:
    foreground = torch.argmax(logits, dim=1)[0].detach().cpu().bool()
    return _one_hot_from_foreground(foreground)


def _label_onehot(batch) -> torch.Tensor:
    foreground = batch["label"][0, 0].detach().cpu().bool()
    return _one_hot_from_foreground(foreground)


def _run_model_inference(
    model,
    mode: str,
    dataloader,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    num_components: int,
    result_root: Path,
    save_outputs: bool,
) -> None:
    method_rows: dict[str, list[dict[str, object]]] = {"LCC": [], "ROI": []}
    method_metrics = {
        "LCC": MetricFactory.create_metrics(),
        "ROI": MetricFactory.create_metrics(),
    }

    model = model.to(device).eval()
    autocast_enabled = device.type == "cuda"
    for batch in tqdm(dataloader, desc=f"Testing {result_root.name}"):
        patient_id = _patient_id(batch)
        image = batch["image"].to(device)

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            if mode == "baseline":
                logits = sliding_window_inference(
                    image,
                    roi_size,
                    sw_batch_size,
                    model.forward,
                )
            else:
                seg = batch["seg"].to(device)
                combined = torch.cat((image, seg), dim=1)
                logits = sliding_window_inference(
                    combined,
                    roi_size,
                    sw_batch_size,
                    lambda window: model(
                        window[:, :1, ...], window[:, 1:, ...]
                    ),
                )

        raw = _raw_prediction(logits)
        label = _label_onehot(batch)
        roi_mask = batch["anatomical_roi"][0, 0].detach().cpu().bool()
        if tuple(roi_mask.shape) != tuple(raw.shape[1:]):
            raise ValueError(
                f"ROI/prediction mismatch for {patient_id}: "
                f"{tuple(roi_mask.shape)} vs {tuple(raw.shape[1:])}"
            )

        predictions = {
            "LCC": keep_largest_components(raw, num_components=num_components),
            "ROI": apply_anatomical_roi(raw, roi_mask),
        }
        target_count = int(label[1].sum().item())
        roi_gt_coverage = (
            float((label[1].bool() & roi_mask).sum().item() / target_count)
            if target_count
            else float("nan")
        )
        affine = _cropped_affine(batch)

        for method, prediction in predictions.items():
            scores = _case_metrics(method_metrics[method], prediction, label)
            removal = _removal_statistics(raw, prediction, label)
            row: dict[str, object] = {
                "dice_score": scores["dice"],
                "hausdorff_score": scores["hausdorff"],
                "iou_score": scores["iou"],
                "precision_score": scores["precision"],
                "recall_score": scores["recall"],
                "cldice_score": scores["cldice"],
                "betti_0_score": scores["betti_0"],
                "betti_1_score": scores["betti_1"],
                "roi_gt_coverage": roi_gt_coverage if method == "ROI" else "",
                **removal,
                "patient_id": patient_id,
            }
            method_rows[method].append(row)

            if save_outputs:
                output_path = (
                    result_root
                    / method
                    / "test"
                    / f"Subj_{patient_id}_outputs.nii.gz"
                )
                _save_output(output_path, prediction, affine)

    for method, rows in method_rows.items():
        result_file = result_root / method / "test" / "test_result.csv"
        _write_result_csv(result_file, rows)
        print(f"Saved {method} results: {result_file}")


def _saved_prediction_paths(prediction_dir: Path, data_dir: Path) -> list[tuple[str, Path]]:
    prefix = "Subj_"
    suffix = "_outputs.nii.gz"
    prediction_by_case: dict[str, Path] = {}
    for path in prediction_dir.glob(f"{prefix}*{suffix}"):
        case_id = path.name[len(prefix) : -len(suffix)]
        if not case_id:
            continue
        if case_id in prediction_by_case:
            raise ValueError(f"duplicate saved prediction for case {case_id}")
        prediction_by_case[case_id] = path

    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test directory not found: {test_dir}")
    case_ids = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
    missing = [case_id for case_id in case_ids if case_id not in prediction_by_case]
    unexpected = sorted(set(prediction_by_case) - set(case_ids))
    if missing or unexpected:
        raise FileNotFoundError(
            "saved prediction/test case mismatch: "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"unexpected={unexpected[:10]} ({len(unexpected)} total)"
        )
    if not case_ids:
        raise RuntimeError(f"no test cases found under {test_dir}")
    return [(case_id, prediction_by_case[case_id]) for case_id in case_ids]


def _load_spatial_nifti(path: Path, description: str):
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{description} must be 3-D, got {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"{description} contains NaN or infinite values: {path}")
    return data, image


def _run_saved_prediction_postprocessing(
    prediction_dir: Path,
    data_dir: Path,
    anatomy_dir: Path,
    tau_mm: float,
    num_components: int,
    result_root: Path,
    save_outputs: bool,
) -> None:
    """Apply both post-processors to already-saved raw binary predictions."""
    cases = _saved_prediction_paths(prediction_dir, data_dir)
    result_files = {
        method: result_root / method / "test" / "test_result.csv"
        for method in ("LCC", "ROI")
    }
    method_rows = {
        method: _read_result_csv(path) for method, path in result_files.items()
    }
    completed_case_ids = set.intersection(
        *(
            {str(row["patient_id"]) for row in rows}
            for rows in method_rows.values()
        )
    )
    for method in method_rows:
        method_rows[method] = [
            row
            for row in method_rows[method]
            if str(row["patient_id"]) in completed_case_ids
        ]
    pending_cases = [case for case in cases if case[0] not in completed_case_ids]
    if completed_case_ids:
        print(
            f"Resuming {result_root.name}: {len(completed_case_ids)} completed, "
            f"{len(pending_cases)} remaining"
        )

    method_metrics = {
        "LCC": MetricFactory.create_metrics(),
        "ROI": MetricFactory.create_metrics(),
    }
    alignment_transform = AnatomicalROIPostProcessingDataModule(
        mode="baseline",
        tau_mm=tau_mm,
        data_dir=str(data_dir),
        anatomy_dir=str(anatomy_dir),
        num_workers=0,
    )._transform()

    for patient_id, prediction_path in tqdm(
        pending_cases, desc=f"Post-processing saved {result_root.name}"
    ):
        image_path = data_dir / "test" / patient_id / "img.nii.gz"
        label_path = data_dir / "test" / patient_id / "label.nii.gz"
        anatomy_path = (
            anatomy_dir
            / "test"
            / patient_id
            / "heart_combined_no_coronary.nii.gz"
        )
        prediction_data, prediction_image = _load_spatial_nifti(
            prediction_path, "prediction"
        )
        aligned = alignment_transform(
            {
                "image": str(image_path),
                "label": str(label_path),
                "anatomy": str(anatomy_path),
            }
        )
        label_data = aligned["label"][0].detach().cpu().numpy()
        roi_mask = aligned["anatomical_roi"][0].detach().cpu().bool()
        aligned_affine = aligned["image"].affine
        if isinstance(aligned_affine, torch.Tensor):
            aligned_affine = aligned_affine.detach().cpu().numpy()

        if prediction_data.shape != label_data.shape:
            raise ValueError(
                f"shape mismatch after reproducing the inference crop for {patient_id}: "
                f"prediction={prediction_data.shape}, aligned_label={label_data.shape}"
            )
        if not np.allclose(
            nib.affines.voxel_sizes(prediction_image.affine),
            nib.affines.voxel_sizes(aligned_affine),
        ):
            raise ValueError(f"voxel spacing mismatch for case {patient_id}")
        prediction_values = set(np.unique(prediction_data).tolist())
        if not prediction_values <= {0, 1}:
            raise ValueError(
                f"saved prediction must be binary for {patient_id}, "
                f"got values {sorted(prediction_values)[:10]}"
            )

        raw = _one_hot_from_foreground(torch.from_numpy(prediction_data > 0))
        label = _one_hot_from_foreground(torch.from_numpy(label_data > 0))
        predictions = {
            "LCC": keep_largest_components(raw, num_components=num_components),
            "ROI": apply_anatomical_roi(raw, roi_mask),
        }
        target_count = int(label[1].sum().item())
        roi_gt_coverage = (
            float((label[1].bool() & roi_mask).sum().item() / target_count)
            if target_count
            else float("nan")
        )

        for method, prediction in predictions.items():
            scores = _case_metrics(method_metrics[method], prediction, label)
            removal = _removal_statistics(raw, prediction, label)
            method_rows[method].append(
                {
                    "dice_score": scores["dice"],
                    "hausdorff_score": scores["hausdorff"],
                    "iou_score": scores["iou"],
                    "precision_score": scores["precision"],
                    "recall_score": scores["recall"],
                    "cldice_score": scores["cldice"],
                    "betti_0_score": scores["betti_0"],
                    "betti_1_score": scores["betti_1"],
                    "roi_gt_coverage": roi_gt_coverage if method == "ROI" else "",
                    **removal,
                    "patient_id": patient_id,
                }
            )
            if save_outputs:
                _save_output(
                    result_root
                    / method
                    / "test"
                    / f"Subj_{patient_id}_outputs.nii.gz",
                    prediction,
                    aligned_affine,
                )

        # Persist every completed case. Atomic replacement prevents a partial
        # CSV if the process is interrupted while writing.
        for method, rows in method_rows.items():
            _write_result_csv(result_files[method], rows)

    for method, rows in method_rows.items():
        result_file = result_files[method]
        _write_result_csv(result_file, rows)
        print(f"Saved {method} results: {result_file}")


@click.command()
@click.option(
    "--model",
    "model_selection",
    type=click.Choice(["baseline", "proposed", "both"]),
    default="both",
    show_default=True,
    help="Select which model(s) to evaluate.",
)
@click.option(
    "--baseline_prediction_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Directory containing saved baseline Subj_*_outputs.nii.gz files.",
)
@click.option(
    "--proposed_prediction_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Directory containing saved proposed Subj_*_outputs.nii.gz files.",
)
@click.option(
    "--baseline_checkpoint",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
)
@click.option(
    "--proposed_checkpoint",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
)
@click.option("--arch_name", type=str, default="SegResNet", show_default=True)
@click.option("--loss_fn", type=str, default="DiceFocalLoss", show_default=True)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="distanceMap",
    show_default=True,
)
@click.option(
    "--data_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("data/imageCAS"),
    show_default=True,
)
@click.option(
    "--anatomy_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("data/imageCAS_no_coronary_conditioning"),
    show_default=True,
)
@click.option(
    "--result_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("result/postprocessing"),
    show_default=True,
)
@click.option("--coverage", type=float, default=0.995, show_default=True)
@click.option("--round_up_mm", type=float, default=0.5, show_default=True)
@click.option("--num_components", type=int, default=2, show_default=True)
@click.option("--gpu_number", type=int, default=0, show_default=True)
@click.option("--num_workers", type=int, default=4, show_default=True)
@click.option("--sw_batch_size", type=int, default=4, show_default=True)
@click.option("--save_outputs", is_flag=True, help="Save case prediction NIfTIs.")
@click.option("--recalibrate", is_flag=True, help="Recompute tau from validation GT.")
def main(
    model_selection: str,
    baseline_prediction_dir: Path | None,
    proposed_prediction_dir: Path | None,
    baseline_checkpoint: Path | None,
    proposed_checkpoint: Path | None,
    arch_name: str,
    loss_fn: str,
    guide: str,
    data_dir: Path,
    anatomy_dir: Path,
    result_dir: Path,
    coverage: float,
    round_up_mm: float,
    num_components: int,
    gpu_number: int,
    num_workers: int,
    sw_batch_size: int,
    save_outputs: bool,
    recalibrate: bool,
) -> None:
    if num_components < 1:
        raise click.ClickException("--num_components must be at least 1")
    if model_selection in {"baseline", "both"} and (
        (baseline_checkpoint is None) == (baseline_prediction_dir is None)
    ):
        raise click.ClickException(
            "select exactly one baseline source: --baseline_checkpoint or "
            "--baseline_prediction_dir"
        )
    if model_selection in {"proposed", "both"} and (
        (proposed_checkpoint is None) == (proposed_prediction_dir is None)
    ):
        raise click.ClickException(
            "select exactly one proposed source: --proposed_checkpoint or "
            "--proposed_prediction_dir"
        )

    needs_inference = (
        model_selection in {"baseline", "both"}
        and baseline_prediction_dir is None
    ) or (
        model_selection in {"proposed", "both"}
        and proposed_prediction_dir is None
    )
    if needs_inference and not torch.cuda.is_available():
        raise click.ClickException("CUDA is required for full-volume model inference")

    device = None
    if needs_inference:
        device = torch.device(f"cuda:{gpu_number}")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("medium")

    result_dir.mkdir(parents=True, exist_ok=True)
    coverage_name = f"{coverage:g}"
    threshold_path = result_dir / f"roi_threshold_coverage_{coverage_name}.json"
    threshold_config = load_or_calibrate_threshold(
        config_path=threshold_path,
        data_dir=data_dir,
        anatomy_dir=anatomy_dir,
        coverage_target=coverage,
        round_up_mm=round_up_mm,
        recalibrate=recalibrate,
    )
    tau_mm = float(threshold_config["tau_mm"])
    print(f"Using anatomical ROI threshold: tau={tau_mm:.3f} mm")

    if model_selection in {"baseline", "both"}:
        baseline_result_root = result_dir / arch_name
        if baseline_prediction_dir is not None:
            print(f"Loading saved baseline predictions: {baseline_prediction_dir}")
            _run_saved_prediction_postprocessing(
                prediction_dir=baseline_prediction_dir,
                data_dir=data_dir,
                anatomy_dir=anatomy_dir,
                tau_mm=tau_mm,
                num_components=num_components,
                result_root=baseline_result_root,
                save_outputs=save_outputs,
            )
        else:
            baseline_data = AnatomicalROIPostProcessingDataModule(
                mode="baseline",
                tau_mm=tau_mm,
                data_dir=str(data_dir),
                anatomy_dir=str(anatomy_dir),
                guide=guide,
                num_workers=num_workers,
            )
            baseline_data.setup("test")
            baseline_model = BaselineSegmentModel.load_from_checkpoint(
                str(baseline_checkpoint),
                map_location="cpu",
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1,
            )
            _run_model_inference(
                model=baseline_model,
                mode="baseline",
                dataloader=baseline_data.test_dataloader(),
                device=device,
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                num_components=num_components,
                result_root=baseline_result_root,
                save_outputs=save_outputs,
            )
            del baseline_model
            torch.cuda.empty_cache()

    if model_selection in {"proposed", "both"}:
        guide_name = "dstMap" if guide == "distanceMap" else "segMap"
        proposed_result_name = f"proposed_{arch_name}_{guide_name}_no_coronary"
        proposed_result_root = result_dir / proposed_result_name
        if proposed_prediction_dir is not None:
            print(f"Loading saved proposed predictions: {proposed_prediction_dir}")
            _run_saved_prediction_postprocessing(
                prediction_dir=proposed_prediction_dir,
                data_dir=data_dir,
                anatomy_dir=anatomy_dir,
                tau_mm=tau_mm,
                num_components=num_components,
                result_root=proposed_result_root,
                save_outputs=save_outputs,
            )
        else:
            proposed_data = AnatomicalROIPostProcessingDataModule(
                mode="proposed_no_coronary",
                tau_mm=tau_mm,
                data_dir=str(data_dir),
                anatomy_dir=str(anatomy_dir),
                guide=guide,
                num_workers=num_workers,
            )
            proposed_data.setup("test")
            proposed_model = CoronaryArteryNoCoronarySegmentModel.load_from_checkpoint(
                str(proposed_checkpoint),
                map_location="cpu",
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1,
                label_nc=7,
            )
            _run_model_inference(
                model=proposed_model,
                mode="proposed_no_coronary",
                dataloader=proposed_data.test_dataloader(),
                device=device,
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                num_components=num_components,
                result_root=proposed_result_root,
                save_outputs=save_outputs,
            )

    print(f"Threshold config: {threshold_path}")
    print(f"All results saved under: {result_dir}")


if __name__ == "__main__":
    main()
