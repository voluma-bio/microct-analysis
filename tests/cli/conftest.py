from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import KDTree

from microct_analysis.processing.femoral_frame import build_femoral_frame
from microct_analysis.processing.mesh_cleanup import preprocess_femoral_mesh
from microct_analysis.processing.surface import extract_surface_mesh

FIXTURES = Path(__file__).parent.parent / "fixtures"
SPACING = np.array([0.0105, 0.0105, 0.0105])
NOTCH_VOXEL = np.array([378.0, 408.0, 306.0])


@pytest.fixture(scope="module")
def oa6_femur_context() -> dict[str, Any]:
    import pyvista as pv

    data = np.load(FIXTURES / "oa6_1rk_femur_condylar.npz")
    vertices, faces = extract_surface_mesh(data["mask"], spacing=(1.0, 1.0, 1.0))
    vertices[:, 0] += int(data["z_offset"])
    vertices[:, 1] += int(data["y_offset"])
    vertices[:, 2] += int(data["x_offset"])
    voxel_vertices, cleaned_faces = preprocess_femoral_mesh(vertices, faces)
    physical_vertices = voxel_vertices * SPACING
    render_mesh = pv.PolyData(
        physical_vertices[:, ::-1],
        np.column_stack([np.full(len(cleaned_faces), 3), cleaned_faces]).ravel(),
    )

    golden = json.loads((FIXTURES / "oa6_1rk_golden.json").read_text())
    groove_voxel = np.asarray(golden["notch"]["groove_position"], dtype=float)
    si_band = voxel_vertices[
        (voxel_vertices[:, 0] >= NOTCH_VOXEL[0] - 18.0)
        & (voxel_vertices[:, 0] <= NOTCH_VOXEL[0] + 52.0)
    ]
    lateral_voxel = si_band[int(np.argmax(si_band[:, 2]))]
    medial_voxel = si_band[int(np.argmin(si_band[:, 2]))]
    landmarks = [
        _landmark("lateral_condylar_edge", lateral_voxel),
        _landmark("medial_condylar_edge", medial_voxel),
        _landmark("intercondylar_groove_midpoint", groove_voxel),
        _landmark("intercondylar_notch", NOTCH_VOXEL),
    ]
    frame = build_femoral_frame(landmarks, physical_vertices)
    assert frame is not None
    return {
        "voxel_vertices": voxel_vertices,
        "physical_vertices": physical_vertices,
        "render_mesh": render_mesh,
        "snap_kdtree": KDTree(physical_vertices),
        "landmarks": landmarks,
        "placed": {"landmarks": landmarks},
        "frame": frame,
        "groove_voxel": groove_voxel,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def landmark(landmark_id: str, voxel: np.ndarray) -> dict[str, Any]:
    return {"id": landmark_id, "voxel": voxel.tolist(), "physical": (voxel * SPACING).tolist()}


def get_landmark(landmarks: list[dict[str, Any]], landmark_id: str) -> dict[str, Any]:
    return next(item for item in landmarks if item["id"] == landmark_id)


def _landmark(landmark_id: str, voxel: np.ndarray) -> dict[str, Any]:
    return landmark(landmark_id, voxel)
