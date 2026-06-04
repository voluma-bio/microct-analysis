from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from microct_analysis.stages.intake import run_intake
from microct_analysis.stages.landmarks_orientation import run_landmarks_orientation
from microct_analysis.stages.measurement import run_measurement
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
    dataset.ManufacturerModelName = "PipelineTest"
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


def _synthetic_labels() -> np.ndarray:
    labels = np.zeros((10, 12, 12), dtype=np.uint16)
    labels[1:5, 2:6, 2:6] = 1
    labels[4:9, 7:10, 7:10] = 2
    return labels


def _write_dicom_series(path: Path, labels: np.ndarray) -> None:
    path.mkdir(parents=True)
    intensity = (labels > 0).astype(np.uint16) * 500
    for z_index, pixels in enumerate(intensity, start=1):
        _write_slice(path / f"slice_{z_index:03d}.dcm", instance=z_index, z=float(z_index - 1), pixels=pixels)


def test_synthetic_volume_runs_full_pipeline(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    labels = _synthetic_labels()
    _write_dicom_series(dicom_dir, labels)

    intake = run_intake(str(dicom_dir), output_dir=str(tmp_path))
    metadata_path = tmp_path / "intake" / "volume_metadata.json"
    assert metadata_path.exists()
    assert intake["spacing"] == [1.0, 1.0, 1.0]

    segmentation = run_segmentation(
        volume=(labels > 0).astype(np.float32) * 500,
        spacing=(1.0, 1.0, 1.0),
        thresholds={"bone_soft_tissue": 200, "subchondral_cortical": 300},
        workflow_thresholds={"labels": labels, "bone_soft_tissue": 200, "tolerance_fraction": 0.5},
        output_dir=str(tmp_path / "segmentation"),
    )
    assert segmentation["status"] == "ready"
    assert Path(segmentation["artifacts"]["filtered"]).exists()
    assignments = json.loads((tmp_path / "segmentation" / "structure_assignments.json").read_text())
    assert assignments["assignments"] == {"femur": 1, "tibia": 2}

    landmark_report = run_landmarks_orientation(
        {
            "labels": segmentation["artifacts"]["labels"],
            "filtered": segmentation["artifacts"]["filtered"],
            "structure_assignments": str(tmp_path / "segmentation" / "structure_assignments.json"),
            "spacing": [1.0, 1.0, 1.0],
        },
        [
            {"id": "femur_center", "structure": "femur", "domain": "femoral_3d_surface", "method": "centroid"},
            {"id": "tibia_center", "structure": "tibia", "domain": "tibial_2d_slice", "method": "centroid"},
        ],
        {"origin_landmark": "femur_center", "axes": {"superior_inferior": {"from": "femur_center", "to": "tibia_center"}}, "target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )
    assert landmark_report["artifacts"]["positions"].endswith("positions.json")
    positions = json.loads((tmp_path / "landmarks" / "positions.json").read_text())
    assert {item["domain"] for item in positions["landmarks"]} == {"femoral_3d_surface", "tibial_2d_slice"}

    measurement_report = run_measurement(
        landmark_artifacts={"positions": str(tmp_path / "landmarks" / "positions.json")},
        roi_artifacts={},
        segmentation_artifacts={"labels": segmentation["artifacts"]["labels"]},
        workflow_measurements=[
            {
                "workflow_id": "synthetic-knee",
                "session_id": "e2e-session",
                "name": "femur_to_tibia_distance",
                "kind": "distance",
                "points": ["femur_center", "tibia_center"],
                "units": "mm",
            },
            {"name": "femur_volume", "kind": "volume", "label_index": 1, "units": "mm^3"},
        ],
        workflow_roi_defs=[],
        spacing=(1.0, 1.0, 1.0),
        output_dir=str(tmp_path / "measurements"),
    )

    results = json.loads((tmp_path / "measurements" / "results.json").read_text())
    assert measurement_report["confidence"] == "high"
    assert results["workflow_id"] == "synthetic-knee"
    assert results["session_id"] == "e2e-session"
    assert {item["name"] for item in results["results"]} == {"femur_to_tibia_distance", "femur_volume"}
    for result in results["results"]:
        assert {"name", "value", "unit", "kind", "spec", "inputs", "qc_evidence"}.issubset(result)
