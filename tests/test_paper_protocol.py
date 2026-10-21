"""Numerical contracts for the standalone paper-protocol reference implementation."""

from pathlib import Path
import sys

import numpy as np
import pydicom
import pytest

REFERENCE = Path(__file__).resolve().parents[1] / "skills/bone-morphometry/resources/python-protocol"
sys.path.insert(0, str(REFERENCE))
from scan import calibration  # noqa: E402
from workflow import bound_array, measure, physical_distance, reslice, save_bound_array, seed_markers  # noqa: E402


def test_implicit_hu_and_private_scanco_calibration_agree():
    ds = pydicom.Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.RescaleSlope = 0.391063
    ds.RescaleIntercept = -1000
    block = ds.private_block(0x29, "SCANCO calibration", create=True)
    block.add_new(0, "IS", 4096)
    block.add_new(6, "DS", 0.62430)
    assert calibration(ds)["private_mapping_consistent"]
    ds.RescaleSlope = 1
    with pytest.raises(ValueError, match="contradicts"):
        calibration(ds)
    ds.RescaleSlope = 0.391063
    ds.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
    with pytest.raises(ValueError, match="No valid"):
        calibration(ds)


def test_physical_distance_handles_anisotropic_rotated_grid():
    affine = np.array([[0, -2, 0, 100], [1, 0, 0, -50], [0, 0, 3, 200], [0, 0, 0, 1]])
    assert physical_distance([[0, 0, 0], [3, 2, 0]], affine) == pytest.approx(5)


def test_slice_interval_and_roi_denominator(tmp_path):
    meta = dict(grid_id="test-grid", shape_zyx=[8, 8, 8], affine_xyz_to_world_mm=np.diag([0.01, 0.01, 0.01, 1]).tolist())
    roi = np.zeros((8, 8, 8), dtype=bool)
    roi[:2, :2, :2] = True
    save_bound_array(tmp_path / "roi.npy", roi, meta)
    hu = np.zeros_like(roi, dtype=float)
    hu[0, :2, :2] = 4000
    spec = dict(grid_id="test-grid", slice_heights={"height": dict(axis_xyz=2, start_index=1, end_index=6,
                                            orientation_reviewed=True, definition="test")},
                regions={"roi": dict(mask_npy=str(tmp_path / "roi.npy"), threshold_hu=2500, definition="test")})
    result = measure(spec, meta, hu=hu)
    assert result["height"]["value"] == pytest.approx(0.05)
    assert result["roi"]["bv_tv_percent"] == 50
    assert result["roi"]["tv_mm3"] == pytest.approx(8e-6)
    with pytest.raises(ValueError, match="identity mismatch"):
        bound_array(tmp_path / "roi.npy", dict(meta, grid_id="different-grid"))
    np.save(tmp_path / "roi.npy", ~roi)
    with pytest.raises(ValueError, match="identity mismatch"):
        bound_array(tmp_path / "roi.npy", meta)


def test_seed_regions_cannot_silently_merge_bones():
    hu = np.full((5, 5, 5), 4000, dtype=float)
    seeds = [dict(label=1, points_zyx=[[1, 1, 1]]), dict(label=2, points_zyx=[[3, 3, 3]])]
    with pytest.raises(ValueError, match="share a connected"):
        seed_markers(hu, seeds)
    hu[2, :, :] = 0
    markers = seed_markers(hu, seeds)
    assert markers[1, 1, 1] == 1
    assert markers[3, 3, 3] == 2


def test_reslice_identity_preserves_voxel_values_and_world_origin():
    hu = np.arange(4*5*6, dtype=np.float32).reshape(4, 5, 6)
    affine = np.diag([0.5, 0.5, 0.5, 1.])
    affine[:3, 3] = [1, 2, 3]
    meta = dict(sources=[], shape_zyx=list(hu.shape), affine_xyz_to_world_mm=affine.tolist())
    result, updated = reslice(hu, meta, np.eye(3))
    np.testing.assert_allclose(result, hu)
    np.testing.assert_allclose(updated["affine_xyz_to_world_mm"], affine)
