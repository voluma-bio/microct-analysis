from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from microct_analysis.stages.measurement import _load_roi_masks
from microct_analysis.stages.roi import compute_roi_boundary, run_roi


def test_compute_roi_boundary_applies_growth_plate_relative_offsets() -> None:
    roi = compute_roi_boundary(
        {
            "id": "proximal_tibia_trabecular",
            "growth_plate_landmark": "tibia_growth_plate",
            "growth_plate_offsets_um": {"z": 50, "y": -100, "x": -200},
            "size_um": {"z": 500, "y": 400, "x": 300},
        },
        {"tibia_growth_plate": {"voxel": [10, 20, 30], "physical": [100, 200, 300]}},
        (10.0, 10.0, 10.0),
    )

    assert roi["positioning"] == "growth-plate-relative"
    assert roi["bounds_physical"] == [[150.0, 650.0], [100.0, 500.0], [100.0, 400.0]]
    assert roi["bounds_voxel"] == [[15.0, 65.0], [10.0, 50.0], [10.0, 40.0]]


def test_run_roi_writes_definitions_masks_and_overlay_contract(tmp_path: Path) -> None:
    landmark_dir = tmp_path / "landmarks"
    landmark_dir.mkdir()
    positions_path = landmark_dir / "positions.json"
    frame_path = landmark_dir / "orientation_frame.json"
    positions_path.write_text(
        json.dumps(
            {
                "spacing": [5, 10, 20],
                "landmarks": [{"id": "growth_plate", "voxel": [2, 3, 4], "physical": [10, 30, 80]}],
            }
        )
    )
    frame_path.write_text(json.dumps({"target_plane": "frontal"}))
    labels = np.zeros((30, 30, 30), dtype=np.uint8)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    report = run_roi(
        {"positions": str(positions_path), "orientation_frame": str(frame_path)},
        {"labels": str(labels_path)},
        [{"id": "tibia_roi", "growth_plate_landmark": "growth_plate", "growth_plate_offsets_um": {"z": 25}, "size_um": {"z": 100, "y": 200, "x": 400}}],
        output_dir=str(tmp_path / "roi"),
    )

    payload = json.loads((tmp_path / "roi" / "roi_definitions.json").read_text())
    mask_metadata = json.loads((tmp_path / "roi" / "masks" / "tibia_roi.json").read_text())
    mask_file = Path(mask_metadata["mask_file"])
    mask = np.asarray(nib.load(str(mask_file)).get_fdata(), dtype=bool)
    assert report["confidence"] == "high"
    assert report["artifacts"]["roi_masks"]["tibia_roi"].endswith("roi/masks/tibia_roi.json")
    assert mask_file.name == "tibia_roi.nii.gz"
    assert mask.shape == labels.shape
    assert mask_metadata["bounds"] == {"z": [7.0, 27.0], "y": [3.0, 23.0], "x": [4.0, 24.0]}
    assert payload["overlay"]["scene"] == "persistent"
    assert payload["rois"][0]["bounds_voxel"] == [[7.0, 27.0], [3.0, 23.0], [4.0, 24.0]]


def test_measurement_loads_roi_mask_metadata_nifti(tmp_path: Path) -> None:
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask_path = tmp_path / "roi_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
    metadata_path = tmp_path / "roi_mask.json"
    metadata_path.write_text(json.dumps({"mask_file": str(mask_path)}))

    masks = _load_roi_masks({"roi_masks": {"trabecular_roi": str(metadata_path)}})

    assert masks["trabecular_roi"].dtype == bool
    assert np.array_equal(masks["trabecular_roi"], mask.astype(bool))
