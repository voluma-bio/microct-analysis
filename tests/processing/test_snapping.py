"""Tests for processing.snapping — pick-to-coordinate snap."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import KDTree

from microct_analysis.processing.snapping import snap_to_slice, snap_to_surface


# ---------------------------------------------------------------------------
# snap_to_slice tests
# ---------------------------------------------------------------------------


def test_snap_to_slice_center_returns_mask_centroid():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[5, 8:12, 6:14] = True  # block centered at y=9.5, x=9.5
    result = snap_to_slice(5, (10, 10), mask, (1.0, 1.0, 1.0), snap_mode="center")
    assert result["coordinate"][0] == 5.0
    assert abs(result["coordinate"][1] - 9.5) < 0.1
    assert abs(result["coordinate"][2] - 9.5) < 0.1
    assert result["on_mask"] is True


def test_snap_to_slice_edge_medial_returns_min_ml():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[5, 8:12, 6:14] = True
    result = snap_to_slice(5, (10, 10), mask, (1.0, 1.0, 1.0), snap_mode="edge_medial")
    assert result["coordinate"][0] == 5.0
    assert result["coordinate"][2] == 6.0  # min x


def test_snap_to_slice_edge_lateral_returns_max_ml():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[5, 8:12, 6:14] = True
    result = snap_to_slice(
        5, (10, 10), mask, (1.0, 1.0, 1.0), snap_mode="edge_lateral"
    )
    assert result["coordinate"][0] == 5.0
    assert result["coordinate"][2] == 13.0  # max x


def test_snap_to_slice_clamps_out_of_range_index():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[:, 8:12, 6:14] = True
    result = snap_to_slice(15, (10, 10), mask, (1.0, 1.0, 1.0), snap_mode="center")
    assert result["coordinate"][0] == 9.0  # clamped to max valid


def test_snap_to_slice_empty_slice_reports_off_mask():
    mask = np.zeros((10, 20, 20), dtype=bool)
    mask[3:7, 8:12, 6:14] = True
    result = snap_to_slice(0, (10, 10), mask, (1.0, 1.0, 1.0), snap_mode="center")
    assert result["on_mask"] is False


# ---------------------------------------------------------------------------
# snap_to_surface tests
# ---------------------------------------------------------------------------

pv = pytest.importorskip("pyvista")


@pytest.mark.skipif(pv is None, reason="pyvista not available")
def test_snap_to_surface_fallback_nearest_to_ray():
    """Construct a scenario where ray_trace misses, triggering the fallback.

    We create a small set of isolated points (no faces) as the render mesh,
    so ray_trace will find no intersection.  The fallback should still pick
    the nearest vertex to the ray line.
    """
    # Vertices in XYZ — a few scattered points
    pts_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )

    # Convert to ZYX for the snap tree
    pts_zyx = pts_xyz[:, ::-1]
    tree = KDTree(pts_zyx)

    # Create a PolyData with no faces — ray_trace will return nothing
    mesh = pv.PolyData(pts_xyz)

    camera_params = {
        "position": [0.0, 0.0, -50.0],
        "focal_point": [0.0, 0.0, 0.0],
        "view_up": [0.0, 1.0, 0.0],
        "view_angle": 30.0,
    }

    # Pick pixel at the centre of the image — ray goes straight through
    # the origin along +Z, so vertex [0,0,0] should be closest.
    result = snap_to_surface(
        pick_pixel=(512, 512),
        camera_params=camera_params,
        render_mesh=mesh,
        snap_kdtree=tree,
        snap_vertices=pts_zyx,
        image_resolution=(1024, 1024),
    )

    assert result["method"] == "nearest_to_ray"
    assert result["intersection_point"] is None
    # The nearest vertex to the ray (along +Z through origin) is [0, 0, 0]
    coord = result["coordinate"]
    assert isinstance(coord, list)
    assert len(coord) == 3
    # Snap distance should be small — the ray passes through the origin
    assert result["snap_distance_mm"] < 1.0
