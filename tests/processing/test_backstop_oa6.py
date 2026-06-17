from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import KDTree

from microct_analysis.processing.backstop import compute_backstop
from microct_analysis.processing.femoral_frame import build_femoral_frame
from microct_analysis.processing.mesh_cleanup import preprocess_femoral_mesh
from microct_analysis.processing.surface import extract_surface_mesh, local_ap_recession

FIXTURES = Path(__file__).parent.parent / "fixtures"
SPACING = np.array([0.0105, 0.0105, 0.0105])
NOTCH_VOXEL = np.array([378.0, 408.0, 306.0])


@pytest.fixture(scope="module")
def oa6_context() -> dict:
    data = np.load(FIXTURES / "oa6_1rk_femur_condylar.npz")
    vertices, faces = extract_surface_mesh(data["mask"], spacing=(1.0, 1.0, 1.0))
    vertices[:, 0] += int(data["z_offset"])
    vertices[:, 1] += int(data["y_offset"])
    vertices[:, 2] += int(data["x_offset"])
    voxel_vertices, _cleaned_faces = preprocess_femoral_mesh(vertices, faces)
    physical_vertices = voxel_vertices * SPACING

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
        "landmarks": landmarks,
        "placed": {"landmarks": landmarks},
        "frame": frame,
        "groove_voxel": groove_voxel,
    }


def test_correct_notch_accepted_high(oa6_context: dict) -> None:
    notch = _get(oa6_context["landmarks"], "intercondylar_notch")

    result = _backstop(oa6_context, "intercondylar_notch", notch["physical"])

    assert result.accepted is True
    assert result.confidence == "high"


def test_anterior_groove_mispick_rejected(oa6_context: dict) -> None:
    groove = _get(oa6_context["landmarks"], "intercondylar_groove_midpoint")

    result = _backstop(oa6_context, "intercondylar_notch", groove["physical"])

    assert result.accepted is False
    assert (
        result.signals["notch_posterior_to_groove"]["accept"] is False
        or result.signals["recession_differential"]["accept"] is False
    )


def test_frame_ap_direction_correct(oa6_context: dict) -> None:
    assert _angle_degrees(oa6_context["frame"].e_AP, np.array([0.375, 0.927, 0.0])) < 15.0


def test_frame_cross_validates_correct_placements(oa6_context: dict) -> None:
    for landmark in oa6_context["landmarks"]:
        result = _backstop(oa6_context, landmark["id"], landmark["physical"])
        assert result.accepted is True, landmark["id"]
        assert result.confidence == "high", landmark["id"]


def test_groove_notch_swap_detected(oa6_context: dict) -> None:
    swapped = [dict(item) for item in oa6_context["landmarks"]]
    groove = _get(swapped, "intercondylar_groove_midpoint")
    notch = _get(swapped, "intercondylar_notch")
    groove["physical"], notch["physical"] = notch["physical"], groove["physical"]
    groove["voxel"], notch["voxel"] = notch["voxel"], groove["voxel"]

    result = compute_backstop(
        {"domain": "femoral_3d_surface", "id": "intercondylar_notch"},
        tuple(notch["physical"]),
        mesh_vertices=oa6_context["physical_vertices"],
        placed_landmarks={"landmarks": swapped},
        femoral_frame=oa6_context["frame"],
    )

    assert result.accepted is False
    assert result.signals["recession_differential"]["accept"] is False


def test_condylar_edges_in_condylar_region(oa6_context: dict) -> None:
    lateral = _get(oa6_context["landmarks"], "lateral_condylar_edge")
    medial = _get(oa6_context["landmarks"], "medial_condylar_edge")

    assert lateral["voxel"][0] > 300.0
    assert medial["voxel"][0] > 300.0


def test_frame_ml_axis_aligned_with_anatomy(oa6_context: dict) -> None:
    assert _angle_degrees(oa6_context["frame"].e_ML, np.array([0.0, 0.0, 1.0])) < 15.0


def test_recession_differential_distinguishes_groove_notch(oa6_context: dict) -> None:
    tree = KDTree(oa6_context["physical_vertices"])
    groove = np.asarray(_get(oa6_context["landmarks"], "intercondylar_groove_midpoint")["physical"])
    notch = np.asarray(_get(oa6_context["landmarks"], "intercondylar_notch")["physical"])

    assert local_ap_recession(notch, oa6_context["physical_vertices"], tree) > local_ap_recession(
        groove, oa6_context["physical_vertices"], tree
    )


def _backstop(context: dict, landmark_id: str, physical: list[float]):
    return compute_backstop(
        {"domain": "femoral_3d_surface", "id": landmark_id},
        tuple(physical),
        mesh_vertices=context["physical_vertices"],
        placed_landmarks=context["placed"],
        femoral_frame=context["frame"],
    )


def _landmark(landmark_id: str, voxel: np.ndarray) -> dict:
    return {"id": landmark_id, "voxel": voxel.tolist(), "physical": (voxel * SPACING).tolist()}


def _get(landmarks: list[dict], landmark_id: str) -> dict:
    return next(item for item in landmarks if item["id"] == landmark_id)


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))
