from __future__ import annotations

import ast
import json
from pathlib import Path

from dataclasses import dataclass

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from microct_analysis.processing.types import Thresholds
from microct_analysis.stages import segmentation


@dataclass(frozen=True)
class Component:
    index: int
    voxel_count: int
    centroid_zyx: tuple[float, float, float]
    centroid_mm: tuple[float, float, float]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    edge_faces: tuple[str, ...]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_segmentation_driver_has_no_mouse_ct_imports(microct_src: Path) -> None:
    imports = [name for name in _imports(microct_src / "stages" / "segmentation.py") if name.startswith("mouse_ct")]
    assert imports == []


def test_stage_report_structure(tmp_path: Path) -> None:
    report = segmentation._report(
        status="ready",
        confidence="high",
        evidence="ok",
        output_root=tmp_path / "segmentation",
        flags=[],
        structure_assignments={"femur": 1},
        threshold_observations=[],
        confounders=[],
    )

    assert report["stage"] == "segmentation"
    assert report["status"] == "ready"
    assert report["recommended_action"] == "proceed"
    assert report["artifacts"]["labels"].endswith("segmentation/labels.nii.gz")
    assert report["artifacts"]["structure_assignments"].endswith("segmentation/structure_assignments.json")
    assert report["artifacts"]["seeds"].endswith("segmentation/seeds.json")
    assert report["artifacts"]["screenshots"] == ["segmentation/screenshot_001.png"]


def test_threshold_comparison_flags_workflow_discrepancy() -> None:
    observations = segmentation.compare_thresholds(
        Thresholds(bone_soft_tissue=240, subchondral_cortical=410),
        {"mask": {"value": 200.0}, "marker": {"value": 400.0}, "tolerance_fraction": 0.10},
    )

    assert len(observations) == 1
    assert "mask threshold derived" in observations[0]


def test_confounder_detection_flags_boundary_extra_and_bridge_components() -> None:
    components = [
        Component(1, 1000, (5.0, 5.0, 5.0), (0.1, 0.1, 0.1), ((0, 21), (2, 8), (2, 8)), ("z_min",)),
        Component(2, 900, (25.0, 5.0, 5.0), (0.5, 0.1, 0.1), ((20, 28), (2, 8), (2, 8)), ()),
        Component(3, 400, (15.0, 1.0, 5.0), (0.3, 0.02, 0.1), ((14, 18), (0, 2), (4, 7)), ()),
        Component(4, 300, (15.0, 9.0, 5.0), (0.3, 0.18, 0.1), ((14, 18), (8, 10), (4, 7)), ()),
        Component(5, 250, (15.0, 7.0, 9.0), (0.3, 0.14, 0.18), ((14, 18), (6, 8), (8, 10)), ()),
    ]

    observations = segmentation.detect_confounders(components, [], (30, 10, 10))

    assert any("Partial bones" in item for item in observations)
    assert any("Sesamoid" in item for item in observations)
    assert any("Osteophytes" in item for item in observations)


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
    dataset.ManufacturerModelName = "StageTest"
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


def _ambiguous_volume() -> np.ndarray:
    volume = np.zeros((32, 32, 32), dtype=np.uint16)
    volume[2:10, 2:10, 8:16] = 500
    volume[2:10, 20:28, 8:16] = 500
    volume[22:30, 10:18, 10:18] = 500
    return volume


def test_full_dicom_path_writes_ready_artifacts(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    out_dir = tmp_path / "segmentation"
    _write_dicom_series(dicom_dir, _ready_volume())

    report = segmentation.run_segmentation(
        dicom_path=str(dicom_dir),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(out_dir),
    )

    assert report["status"] == "ready"
    assert (out_dir / "labels.nii.gz").exists()
    assert (out_dir / "masks" / "femur.nii.gz").exists()
    assert (out_dir / "stage_report.json").exists()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["status"] == "ready"
    assert metadata["scanner"] == "unknown"
    assert metadata["thresholds"]["method"] == "manual-override"
    assert metadata["components"]["retained"] == 3
    assignments = json.loads((out_dir / "structure_assignments.json").read_text())
    assert assignments["assignments"] == {"femur": 1, "tibia": 2, "patella": 4}
    assert assignments["component_assignments"]["femur"] == 1
    assert assignments["component_assignments"]["tibia"] == 3


def test_full_path_accepts_zero_mask_threshold(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    out_dir = tmp_path / "segmentation"
    volume = _ready_volume()
    volume[volume == 0] = 1
    _write_dicom_series(dicom_dir, volume)

    report = segmentation.run_segmentation(
        dicom_path=str(dicom_dir),
        thresholds={"mask": 0, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(out_dir),
    )

    assert report["status"] == "ready"
    assert json.loads((out_dir / "metadata.json").read_text())["thresholds"]["mask"] == 0.0


def test_full_path_can_load_from_intake_metadata(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    out_dir = tmp_path / "segmentation"
    metadata_path = tmp_path / "volume_metadata.json"
    _write_dicom_series(dicom_dir, _ready_volume())
    metadata_path.write_text(json.dumps({"dicom_path": str(dicom_dir)}), encoding="utf-8")

    report = segmentation.run_segmentation(
        intake_metadata_path=str(metadata_path),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(out_dir),
    )

    assert report["status"] == "ready"
    assert json.loads((out_dir / "metadata.json").read_text())["status"] == "ready"


def test_full_path_resolves_relative_dicom_path_from_intake_metadata(tmp_path: Path) -> None:
    intake_dir = tmp_path / "intake"
    dicom_dir = intake_dir / "dicom"
    out_dir = tmp_path / "segmentation"
    intake_dir.mkdir()
    _write_dicom_series(dicom_dir, _ready_volume())
    metadata_path = intake_dir / "volume_metadata.json"
    metadata_path.write_text(json.dumps({"dicom_path": "dicom"}), encoding="utf-8")

    report = segmentation.run_segmentation(
        intake_metadata_path=str(metadata_path),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(out_dir),
    )

    assert report["status"] == "ready"


def test_ambiguous_full_path_emits_seeds_and_rerun_resolves(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    _write_dicom_series(dicom_dir, _ambiguous_volume())

    first = segmentation.run_segmentation(
        dicom_path=str(dicom_dir),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        output_dir=str(first_out),
    )

    assert first["status"] == "needs-seeds"
    assert not (first_out / "labels.nii.gz").exists()
    assert not (first_out / "masks").exists()
    assert (first_out / "components.nii.gz").exists()
    seeds = json.loads((first_out / "seeds.json").read_text())
    assert set(seeds) == {"assignments", "scan_fingerprint", "schema_version"}
    assert set(seeds["assignments"]) == {"femur", "tibia"}

    second = segmentation.run_segmentation(
        dicom_path=str(dicom_dir),
        thresholds={"mask": 100, "marker": 300},
        min_marker_voxels=20,
        seeds_path=str(first_out / "seeds.json"),
        output_dir=str(second_out),
    )

    assert second["status"] == "ready"
    metadata = json.loads((second_out / "metadata.json").read_text())
    assert metadata["status"] == "ready"
    assert "ambiguity-resolved-via-seeds" in metadata["flags"]
    assert (second_out / "labels.nii.gz").exists()
