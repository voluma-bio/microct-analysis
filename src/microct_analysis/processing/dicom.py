"""DICOM loading for micro-CT scan directories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from microct_analysis.processing.types import ScanVolume


class LoadError(Exception):
    """Raised when a DICOM directory cannot be loaded as a scan volume."""

    def __init__(self, flag: str, message: str | None = None) -> None:
        if message is None:
            message = flag
            flag = "load-error"
        super().__init__(message)
        self.flag = flag


def load_dicom(path: Path, *, command: str = "load_dicom", pipeline_version: str = "microct-analysis") -> ScanVolume:
    """Load a DICOM directory as a float32 ``(z, y, x)`` scan volume.

    Slices are sorted by projecting ``ImagePositionPatient`` onto the slice
    normal derived from ``ImageOrientationPatient``. The returned provenance
    includes stable DICOM fingerprint fields for downstream segmentation seed
    validation.
    """

    scan_dir = Path(path).resolve()
    if not scan_dir.is_dir():
        raise LoadError("missing-dicom-files", f"DICOM path is not a directory: {scan_dir}")

    files = _iter_dicom_files(scan_dir)
    slices: list[tuple[float, FileDataset, Path]] = []
    for file in files:
        try:
            dataset = pydicom.dcmread(str(file))
        except pydicom.errors.InvalidDicomError:
            continue
        except Exception as exc:  # pragma: no cover - pydicom exception types vary by input
            raise LoadError("dicom-read-error", f"Failed to read {file}: {exc}") from exc
        _check_required(dataset, file)
        try:
            orientation = [float(value) for value in dataset.ImageOrientationPatient]
            position = [float(value) for value in dataset.ImagePositionPatient]
        except (TypeError, ValueError) as exc:
            raise LoadError("missing-dicom-tags", f"{file}: invalid image orientation/position tags") from exc
        normal = _slice_normal(orientation)
        projection = _project(position, normal)
        slices.append((projection, dataset, file))

    if not slices:
        raise LoadError("missing-dicom-files", f"No valid CT DICOM files found in directory: {scan_dir}")

    slices.sort(key=lambda item: item[0])
    ordered = [dataset for _, dataset, _ in slices]
    ordered_paths = [file for _, _, file in slices]

    spacing = _spacing(slices)
    data = np.stack([_pixel_array(dataset, file).astype(np.float32, copy=False) for dataset, file in zip(ordered, ordered_paths, strict=True)], axis=0)
    affine = _affine(ordered[0], spacing)
    provenance = _provenance(scan_dir, ordered, spacing, command, pipeline_version)

    return ScanVolume(data=data, spacing=spacing, affine=affine, provenance=provenance)


def _iter_dicom_files(directory: Path) -> list[Path]:
    files = [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in (".dcm", ".ima", "")]
    if not files:
        raise LoadError("missing-dicom-files", f"No DICOM files found in directory: {directory}")
    return files


def _check_required(dataset: FileDataset, path: Path) -> None:
    required = [
        "Modality",
        "PixelSpacing",
        "SliceThickness",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "BitsStored",
        "PixelRepresentation",
    ]
    missing = [tag for tag in required if not hasattr(dataset, tag)]
    if missing:
        raise LoadError("missing-dicom-tags", f"{path}: missing required tags {missing}")
    if str(dataset.Modality) != "CT":
        raise LoadError("wrong-modality", f"{path}: expected Modality=CT, got {dataset.Modality!r}")


def _slice_normal(orientation: list[float]) -> np.ndarray:
    if len(orientation) != 6:
        raise LoadError("missing-dicom-tags", "ImageOrientationPatient must contain six values")
    row = np.array(orientation[:3], dtype=np.float64)
    col = np.array(orientation[3:], dtype=np.float64)
    normal = np.cross(row, col)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise LoadError("missing-dicom-tags", "ImageOrientationPatient row/column vectors are degenerate")
    return normal / norm


def _project(position: list[float], normal: np.ndarray) -> float:
    if len(position) < 3:
        raise LoadError("missing-dicom-tags", "ImagePositionPatient must contain at least three values")
    return float(np.dot(np.array(position[:3], dtype=np.float64), normal))


def _pixel_array(dataset: FileDataset, path: Path) -> np.ndarray:
    try:
        array = dataset.pixel_array
    except Exception as exc:  # pydicom raises different exceptions for codec failures
        file_meta = getattr(dataset, "file_meta", None)
        transfer_syntax = getattr(file_meta, "TransferSyntaxUID", "<missing>")
        raise LoadError(
            "missing-codec",
            f"{path}: cannot decode pixel data; TransferSyntaxUID: {transfer_syntax}. Underlying: {exc!r}",
        ) from exc
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return array.astype(np.float32) * np.float32(slope) + np.float32(intercept)


def _spacing(slices: list[tuple[float, FileDataset, Path]]) -> tuple[float, float, float]:
    first = slices[0][1]
    pixel_spacing = getattr(first, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) < 2:
        raise LoadError("missing-dicom-tags", "DICOM metadata missing PixelSpacing")

    y_spacing = float(pixel_spacing[0])
    x_spacing = float(pixel_spacing[1])
    if len(slices) >= 2:
        projections = [projection for projection, _, _ in slices]
        deltas = np.diff(projections)
        mean_delta = float(np.mean(np.abs(deltas)))
        rel_err = np.abs(np.abs(deltas) - mean_delta) / max(mean_delta, 1e-9)
        if float(np.max(rel_err)) > 0.01:
            raise LoadError("non-uniform-spacing", f"slice spacing varies by >1% (max rel err {float(rel_err.max()):.3%})")
        z_spacing = mean_delta
    else:
        z_spacing = float(first.SliceThickness)
    return (float(z_spacing), y_spacing, x_spacing)


def _affine(first: FileDataset, spacing: tuple[float, float, float]) -> np.ndarray:
    z_spacing, y_spacing, x_spacing = spacing
    orientation = [float(value) for value in first.ImageOrientationPatient]
    row = np.array(orientation[:3], dtype=np.float64) * x_spacing
    col = np.array(orientation[3:], dtype=np.float64) * y_spacing
    normal = _slice_normal(orientation) * z_spacing
    origin = np.array([float(value) for value in first.ImagePositionPatient[:3]], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row
    affine[:3, 1] = col
    affine[:3, 2] = normal
    affine[:3, 3] = origin
    return affine


def _provenance(
    directory: Path,
    datasets: list[FileDataset],
    spacing: tuple[float, float, float],
    command: str,
    pipeline_version: str,
) -> dict[str, Any]:
    first = datasets[0]
    file_meta = getattr(first, "file_meta", None)
    uid_hash = hashlib.sha256(
        b"".join(str(getattr(dataset, "SOPInstanceUID", "")).encode() for dataset in datasets)
    ).hexdigest()[:16]
    manufacturer = str(getattr(first, "Manufacturer", "") or "")
    model = str(getattr(first, "ManufacturerModelName", "") or "")
    transfer_syntax = str(getattr(file_meta, "TransferSyntaxUID", "") or "")
    return {
        "manufacturer": manufacturer,
        "model": model,
        "acquisition_date": str(getattr(first, "AcquisitionDate", "")),
        "slice_count": len(datasets),
        "source_dir": str(directory),
        "n_slices": len(datasets),
        "original_spacing": list(spacing),
        "resampled_spacing": None,
        "slice_uid_hash": uid_hash,
        "pipeline_version": pipeline_version,
        "command": command,
        "manufacturer_raw": manufacturer,
        "model_raw": model,
        "transfer_syntax_uid": transfer_syntax,
    }
