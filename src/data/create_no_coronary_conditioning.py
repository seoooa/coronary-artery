"""Create coronary-free conditioning maps from the imageCAS heart label maps.

The source dataset is never modified. Only the derived conditioning volumes and
their metadata are written to ``output_root``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import click
import nibabel as nib
import numpy as np


SOURCE_FILENAME = "heart_combined.nii.gz"
OUTPUT_FILENAME = "heart_combined_no_coronary.nii.gz"
SPLITS = ("train", "valid", "test")

# Coronary arteries (source label 1) become background. The six remaining
# anatomical structures are made contiguous so MONAI can one-hot encode them
# into seven channels including background.
LABEL_MAPPING = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
}

CLASS_NAMES = {
    0: "background (including removed coronary arteries)",
    1: "aorta",
    2: "heart_myocardium",
    3: "heart_ventricle_left",
    4: "heart_ventricle_right",
    5: "heart_atrium_left",
    6: "heart_atrium_right",
}


def _integer_label_data(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("source contains NaN or infinite values")

    integer_data = data.astype(np.int16)
    if not np.array_equal(data, integer_data):
        raise ValueError("source contains non-integer label values")

    unexpected = sorted(set(np.unique(integer_data).tolist()) - set(LABEL_MAPPING))
    if unexpected:
        raise ValueError(f"unexpected source labels: {unexpected}")
    return integer_data


def _remap(source: np.ndarray) -> np.ndarray:
    lookup = np.asarray([LABEL_MAPPING[index] for index in range(8)], dtype=np.uint8)
    return lookup[source]


def _save_nifti_atomic(
    data: np.ndarray,
    source_image: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = source_image.header.copy()
    header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(data, source_image.affine, header=header)

    temporary_path = output_path.with_name(f".{output_path.name}.tmp.nii.gz")
    nib.save(output_image, temporary_path)
    os.replace(temporary_path, output_path)


def _validate_output(
    output_path: Path,
    source_image: nib.spatialimages.SpatialImage,
    expected_data: np.ndarray,
) -> tuple[bool, bool]:
    output_image = nib.load(output_path)
    output_data = np.asanyarray(output_image.dataobj)

    if not np.array_equal(output_data, expected_data):
        raise ValueError("saved output does not match the expected remapping")

    shape_matches = output_image.shape == source_image.shape
    affine_matches = np.allclose(output_image.affine, source_image.affine)
    if not shape_matches or not affine_matches:
        raise ValueError(
            f"geometry mismatch: shape={shape_matches}, affine={affine_matches}"
        )
    return shape_matches, affine_matches


def _class_counts(data: np.ndarray, labels: range) -> dict[int, int]:
    return {label: int(np.count_nonzero(data == label)) for label in labels}


def _write_class_info(output_root: Path) -> None:
    lines = [
        "No-Coronary Conditioning Class Information",
        "==========================================",
        "",
        "Source label 1 (coronary_arteries) is mapped to background.",
        "",
    ]
    lines.extend(f"Label {label}: {name}" for label, name in CLASS_NAMES.items())
    lines.append("")
    (output_root / "combined_class_info.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "case_id",
        "source_file",
        "output_file",
        "status",
        "shape_matches",
        "affine_matches",
        "removed_coronary_voxels",
    ]
    fieldnames.extend(f"source_label_{label}_voxels" for label in range(8))
    fieldnames.extend(f"output_label_{label}_voxels" for label in range(7))

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_dataset(
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source_root and output_root must be different")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for split in SPLITS:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"dataset split not found: {split_dir}")

        case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        click.echo(f"[{split}] processing {len(case_dirs)} cases")

        for case_dir in case_dirs:
            image_path = case_dir / "img.nii.gz"
            label_path = case_dir / "label.nii.gz"
            source_path = case_dir / SOURCE_FILENAME
            missing = [
                str(path)
                for path in (image_path, label_path, source_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(f"missing files for {split}/{case_dir.name}: {missing}")

            source_image = nib.load(source_path)
            source_data = _integer_label_data(source_image)
            source_counts = _class_counts(source_data, range(8))
            missing_labels = [label for label in range(8) if source_counts[label] == 0]
            if missing_labels:
                raise ValueError(
                    f"{split}/{case_dir.name} is missing source labels: {missing_labels}"
                )

            output_data = _remap(source_data)
            output_counts = _class_counts(output_data, range(7))
            output_path = output_root / split / case_dir.name / OUTPUT_FILENAME

            existed_before = output_path.exists()
            if existed_before and not overwrite:
                status = "validated_existing"
            else:
                _save_nifti_atomic(output_data, source_image, output_path)
                status = "overwritten" if existed_before else "created"

            shape_matches, affine_matches = _validate_output(
                output_path, source_image, output_data
            )

            row: dict[str, object] = {
                "split": split,
                "case_id": case_dir.name,
                "source_file": str(source_path),
                "output_file": str(output_path),
                "status": status,
                "shape_matches": shape_matches,
                "affine_matches": affine_matches,
                "removed_coronary_voxels": source_counts[1],
            }
            row.update(
                {
                    f"source_label_{label}_voxels": source_counts[label]
                    for label in range(8)
                }
            )
            row.update(
                {
                    f"output_label_{label}_voxels": output_counts[label]
                    for label in range(7)
                }
            )
            manifest_rows.append(row)

        click.echo(f"[{split}] completed")

    _write_class_info(output_root)
    _write_manifest(output_root, manifest_rows)
    return manifest_rows


@click.command()
@click.option(
    "--source_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/imageCAS"),
    show_default=True,
)
@click.option(
    "--output_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/imageCAS_no_coronary_conditioning"),
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="Replace existing derived maps.")
def main(source_root: Path, output_root: Path, overwrite: bool) -> None:
    """Generate the complete no-coronary conditioning sidecar dataset."""
    rows = create_dataset(source_root, output_root, overwrite=overwrite)
    click.echo(f"Created and validated {len(rows)} conditioning maps in {output_root}")


if __name__ == "__main__":
    main()
