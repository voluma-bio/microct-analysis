from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from microct_analysis.stages.intake import run_intake
from microct_analysis.stages.segmentation import run_segmentation


def _write_slice(path: Path, *, instance: int, z: float, pixels: np.ndarray) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.Manufacturer = "Synthetic"
    dataset.ManufacturerModelName = "IntakeStageTest"
    dataset.AcquisitionDate = "20260504"
    dataset.InstanceNumber = instance
    dataset.ImagePositionPatient = [0.0, 0.0, z]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.Rows = int(pixels.shape[0])
    dataset.Columns = int(pixels.shape[1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, write_like_original=False)


def _write_dicom_series(path: Path, volume: np.ndarray) -> None:
    path.mkdir(parents=True)
    for z_index, pixels in enumerate(volume, start=1):
        _write_slice(path / f"slice_{z_index:03d}.dcm", instance=z_index, z=float(z_index - 1), pixels=pixels)


def _ready_volume() -> np.ndarray:
    volume = np.zeros((32, 32, 32), dtype=np.uint16)
    volume[2:14, 10:22, 10:22] = 500
    volume[22:30, 10:18, 10:18] = 500
    volume[14:21, 2:9, 12:19] = 500
    return volume


def test_intake_metadata_supports_segmentation_rerun_without_raw_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dicom_series(Path("dicom"), _ready_volume())

    metadata = run_intake("dicom", output_dir="session")

    intake_dir = tmp_path / "session" / "intake"
    metadata_path = intake_dir / "volume_metadata.json"
    assert metadata_path.exists()
    assert (intake_dir / "orientation_report.md").exists()
    assert (intake_dir / "stage_report.json").exists()
    assert {path.name for path in intake_dir.iterdir()} == {
        "orientation_report.md",
        "stage_report.json",
        "volume_metadata.json",
    }

    persisted = json.loads(metadata_path.read_text())
    assert persisted["dicom_path"] == "dicom"
    assert Path(persisted["dicom_source_dir"]).is_absolute()
    assert persisted["original_spacing"] == [1.0, 1.0, 1.0]
    assert persisted["provenance"]["slice_uid_hash"]
    assert persisted["fingerprint"]["slice_uid_hash"] == persisted["provenance"]["slice_uid_hash"]
    assert persisted["scanner_profile"] == "unknown"
    assert persisted["scanner_detection"]["profile"] == "unknown"
    assert "segmentation_threshold_analysis" in persisted
    assert isinstance(persisted["segmentation_ready"], bool)
    assert metadata["dicom_source_dir"] == persisted["dicom_source_dir"]

    stage_report = json.loads((intake_dir / "stage_report.json").read_text())
    assert stage_report["stage"] == "intake"
    assert stage_report["status"] == "ready"
    assert stage_report["recommended_action"] in {"proceed", "flag"}
    assert stage_report["artifacts"]["volume_metadata"].endswith("intake/volume_metadata.json")
    assert stage_report["artifacts"]["stage_report"].endswith("intake/stage_report.json")
    assert stage_report["segmentation_ready"] == persisted["segmentation_ready"]
    assert stage_report["threshold_analysis_status"] == persisted["segmentation_threshold_analysis"]["status"]

    report = run_segmentation(
        intake_metadata_path=str(metadata_path),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(tmp_path / "segmentation"),
    )

    assert report["status"] == "ready"
    segmentation_metadata = json.loads((tmp_path / "segmentation" / "metadata.json").read_text())
    assert segmentation_metadata["status"] == "ready"
    assert segmentation_metadata["provenance"]["source_dir"] == persisted["dicom_source_dir"]


def test_segmentation_metadata_path_resolution_falls_back_to_existing_candidate(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    _write_dicom_series(dicom_dir, _ready_volume())
    run_intake(str(dicom_dir), output_dir=str(tmp_path / "session"))
    metadata_path = tmp_path / "session" / "intake" / "volume_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    stale_path = str(tmp_path / "missing-dicom")
    metadata["dicom_source_dir"] = stale_path
    metadata["provenance"]["source_dir"] = stale_path
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = run_segmentation(
        intake_metadata_path=str(metadata_path),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(tmp_path / "segmentation"),
    )

    assert report["status"] == "ready"
    segmentation_metadata = json.loads((tmp_path / "segmentation" / "metadata.json").read_text())
    assert segmentation_metadata["provenance"]["source_dir"] == str(dicom_dir)
