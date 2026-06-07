"""Tests for processing.rendering module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from scipy.spatial import KDTree

from microct_analysis.processing.rendering import (
    _xyz_to_zyx,
    _zyx_to_xyz,
    query_local_geometry,
    render_slice_view,
    validate_segmentation_for_landmarking,
)


# ---------------------------------------------------------------------------
# test_validate_segmentation_passes_clean_labels
# ---------------------------------------------------------------------------


def test_validate_segmentation_passes_clean_labels() -> None:
    """Two well-separated label regions should pass validation."""
    vol = np.zeros((30, 30, 30), dtype=np.uint8)
    # Femur: solid block in upper-left region (not touching boundaries)
    vol[5:12, 5:12, 5:12] = 1
    # Tibia: solid block in lower-right region (not touching boundaries)
    vol[18:25, 18:25, 18:25] = 2

    assignments = {"femur": 1, "tibia": 2}
    ok, reason = validate_segmentation_for_landmarking(vol, assignments)
    assert ok is True
    assert reason == "ok"


# ---------------------------------------------------------------------------
# test_validate_segmentation_detects_merged_bone
# ---------------------------------------------------------------------------


def test_validate_segmentation_detects_merged_bone() -> None:
    """One label with two disconnected components should fail."""
    vol = np.zeros((30, 30, 30), dtype=np.uint8)
    # Two disconnected blobs with the same label
    vol[3:7, 3:7, 3:7] = 1
    vol[20:24, 20:24, 20:24] = 1

    assignments = {"femur": 1}
    ok, reason = validate_segmentation_for_landmarking(vol, assignments)
    assert ok is False
    assert "connected components" in reason


# ---------------------------------------------------------------------------
# test_validate_segmentation_detects_fov_crop
# ---------------------------------------------------------------------------


def test_validate_segmentation_detects_fov_crop() -> None:
    """Label touching volume boundary on >2 faces should fail."""
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    # A label that spans from face 0 to face -1 on all three axes
    # touches all 6 faces
    vol[0:20, 0:20, 0:20] = 1

    assignments = {"femur": 1}
    ok, reason = validate_segmentation_for_landmarking(vol, assignments)
    assert ok is False
    assert "boundary faces" in reason


# ---------------------------------------------------------------------------
# test_prepare_landmark_session_builds_meshes_and_kdtrees
# ---------------------------------------------------------------------------


def test_prepare_landmark_session_builds_meshes_and_kdtrees(tmp_path) -> None:
    """Verify dict structure with meshes for each bone using synthetic volume."""
    pytest.importorskip("pyvista")
    import nibabel as nib

    # Create a small synthetic volume with two small spheres
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    # Sphere 1 (femur) centred at (5, 10, 10)
    z, y, x = np.indices((20, 20, 20))
    femur_mask = ((z - 5) ** 2 + (y - 10) ** 2 + (x - 10) ** 2) <= 9
    vol[femur_mask] = 1
    # Sphere 2 (tibia) centred at (15, 10, 10)
    tibia_mask = ((z - 15) ** 2 + (y - 10) ** 2 + (x - 10) ** 2) <= 9
    vol[tibia_mask] = 2

    # Save as NIfTI
    labels_path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(labels_path))

    # Save assignments
    assignments_path = tmp_path / "structure_assignments.json"
    import json

    assignments_path.write_text(json.dumps({"femur": 1, "tibia": 2}))

    artifacts = {
        "labels": str(labels_path),
        "structure_assignments": str(assignments_path),
    }

    from microct_analysis.processing.rendering import prepare_landmark_session

    session = prepare_landmark_session(artifacts)

    assert "labels" in session
    assert "intensity" in session
    assert "assignments" in session
    assert "spacing" in session
    assert "meshes" in session

    assert session["assignments"] == {"femur": 1, "tibia": 2}
    assert "femur" in session["meshes"]
    assert "tibia" in session["meshes"]

    for bone_name in ("femur", "tibia"):
        mesh_data = session["meshes"][bone_name]
        assert "render" in mesh_data
        assert "snap_vertices" in mesh_data
        assert "snap_kdtree" in mesh_data
        assert isinstance(mesh_data["snap_vertices"], np.ndarray)
        assert isinstance(mesh_data["snap_kdtree"], KDTree)
        assert mesh_data["snap_vertices"].ndim == 2
        assert mesh_data["snap_vertices"].shape[1] == 3


# ---------------------------------------------------------------------------
# test_render_slice_view_produces_png
# ---------------------------------------------------------------------------


def test_render_slice_view_produces_png(tmp_path) -> None:
    """Verify file is written and is valid PNG."""
    volume = np.random.default_rng(42).random((20, 20, 20)) * 1000
    output = str(tmp_path / "slice.png")

    result = render_slice_view(
        volume,
        slice_axis=0,
        slice_index=10,
        window=(500.0, 800.0),
        output_path=output,
    )

    assert result == output
    assert os.path.exists(result)
    # Check PNG magic bytes
    with open(result, "rb") as f:
        magic = f.read(8)
    assert magic[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# test_query_local_geometry_returns_nearest_vertex
# ---------------------------------------------------------------------------


def test_query_local_geometry_returns_nearest_vertex() -> None:
    """Simple geometry check with known vertices."""
    # Create a simple grid of vertices in ZYX
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    tree = KDTree(vertices)

    # Query a point close to vertex [1, 0, 0]
    result = query_local_geometry((1.1, 0.05, 0.05), vertices, tree, k=6)

    assert "nearest_vertex" in result
    assert "distance_mm" in result
    assert "neighbors" in result
    assert "local_normal" in result
    assert "local_curvature" in result

    # Nearest should be [1.0, 0.0, 0.0]
    nearest = np.array(result["nearest_vertex"])
    assert np.allclose(nearest, [1.0, 0.0, 0.0])
    assert result["distance_mm"] < 0.2  # very close


# ---------------------------------------------------------------------------
# test_zyx_xyz_conversion_roundtrip
# ---------------------------------------------------------------------------


def test_zyx_xyz_conversion_roundtrip() -> None:
    """Verify _zyx_to_xyz and _xyz_to_zyx are inverse operations."""
    rng = np.random.default_rng(123)

    # Test with 2D array (N, 3)
    coords_2d = rng.random((50, 3)) * 100
    roundtrip_2d = _xyz_to_zyx(_zyx_to_xyz(coords_2d))
    assert np.allclose(roundtrip_2d, coords_2d)

    # Test with 1D array (3,)
    coords_1d = rng.random(3) * 100
    roundtrip_1d = _xyz_to_zyx(_zyx_to_xyz(coords_1d))
    assert np.allclose(roundtrip_1d, coords_1d)

    # Verify actual conversion: ZYX [z, y, x] -> XYZ [x, y, z]
    zyx = np.array([[1.0, 2.0, 3.0]])
    xyz = _zyx_to_xyz(zyx)
    assert np.allclose(xyz, [[3.0, 2.0, 1.0]])
