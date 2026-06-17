from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microct_analysis.processing.femoral_frame import FemoralFrame, build_femoral_frame
from microct_analysis.processing.mesh_cleanup import preprocess_femoral_mesh
from microct_analysis.processing.surface import condylar_region_mask, extract_surface_mesh

FIXTURES = Path(__file__).parent.parent / "fixtures"
SPACING = np.array([0.0105, 0.0105, 0.0105])


@pytest.fixture(scope="module")
def oa6_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(FIXTURES / "oa6_1rk_femur_condylar.npz")
    vertices, faces = extract_surface_mesh(data["mask"], spacing=(1.0, 1.0, 1.0))
    vertices[:, 0] += int(data["z_offset"])
    vertices[:, 1] += int(data["y_offset"])
    vertices[:, 2] += int(data["x_offset"])
    cleaned_vertices, cleaned_faces = preprocess_femoral_mesh(vertices, faces)
    assert cleaned_faces is not None
    return cleaned_vertices, cleaned_faces, cleaned_vertices * SPACING


@pytest.fixture(scope="module")
def oa6_landmarks(oa6_mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[dict]:
    voxel_vertices, _faces, _physical_vertices = oa6_mesh
    golden = json.loads((FIXTURES / "oa6_1rk_golden.json").read_text())

    # The fixture does not store visual L/M picks; synthesize them from the
    # same mesh-level condylar region used by the backstop.
    notch_voxel = np.array([378.0, 408.0, 306.0])
    si_band = voxel_vertices[
        (voxel_vertices[:, 0] >= notch_voxel[0] - 18.0) & (voxel_vertices[:, 0] <= notch_voxel[0] + 52.0)
    ]
    lateral = si_band[int(np.argmax(si_band[:, 2]))] * SPACING
    medial = si_band[int(np.argmin(si_band[:, 2]))] * SPACING
    groove = np.asarray(golden["notch"]["groove_position"], dtype=float) * SPACING
    notch = notch_voxel * SPACING
    return [
        {"id": "lateral_condylar_edge", "physical": lateral.tolist()},
        {"id": "medial_condylar_edge", "physical": medial.tolist()},
        {"id": "intercondylar_groove_midpoint", "physical": groove.tolist()},
        {"id": "intercondylar_notch", "physical": notch.tolist()},
    ]


def test_preprocess_femoral_mesh_keeps_largest_component() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [11.0, 10.0, 10.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int32)

    cleaned_vertices, cleaned_faces = preprocess_femoral_mesh(vertices, faces)

    assert cleaned_vertices.shape == (4, 3)
    assert cleaned_faces is not None
    assert cleaned_faces.shape == (2, 3)
    assert np.all(cleaned_vertices[:, 0] < 2.0)
    assert cleaned_faces.max() == 3


def test_oa6_fixture_is_real_condylar_mesh(oa6_mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    voxel_vertices, faces, _physical_vertices = oa6_mesh

    assert voxel_vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert 1_000_000 < len(voxel_vertices) < 1_800_000
    assert np.allclose(np.ptp(voxel_vertices, axis=0), [468.0, 328.0, 370.0])


def test_build_femoral_frame_on_real_oa6_mesh(oa6_mesh: tuple[np.ndarray, np.ndarray, np.ndarray], oa6_landmarks: list[dict]) -> None:
    _voxel_vertices, _faces, physical_vertices = oa6_mesh

    frame = build_femoral_frame(oa6_landmarks, physical_vertices)

    assert isinstance(frame, FemoralFrame)
    assert frame.confidence == "high"
    ml_angle_to_x = _angle_degrees(frame.e_ML, np.array([0.0, 0.0, 1.0]))
    assert ml_angle_to_x < 15.0
    assert frame.evidence["ap_verification"] == "mesh_density_confirmed"
    assert frame.evidence["ap_density_ratio"] > frame.evidence["density_ratio_threshold"]
    assert float(np.dot(frame.e_AP, np.array([0.0, 1.0, 0.0]))) < 0.0
    assert np.linalg.det(np.stack([frame.e_ML, frame.e_AP, frame.e_SI], axis=0)) == pytest.approx(1.0)


def test_groove_notch_swap_flips_back_to_same_ap_direction(
    oa6_mesh: tuple[np.ndarray, np.ndarray, np.ndarray], oa6_landmarks: list[dict]
) -> None:
    _voxel_vertices, _faces, physical_vertices = oa6_mesh
    expected = build_femoral_frame(oa6_landmarks, physical_vertices)
    swapped = [dict(item) for item in oa6_landmarks]
    groove = next(item for item in swapped if item["id"] == "intercondylar_groove_midpoint")
    notch = next(item for item in swapped if item["id"] == "intercondylar_notch")
    groove["physical"], notch["physical"] = notch["physical"], groove["physical"]

    frame = build_femoral_frame(swapped, physical_vertices)

    assert expected is not None
    assert frame is not None
    assert frame.confidence == "medium"
    assert frame.evidence["ap_verification"] == "mesh_density_flipped"
    assert float(np.dot(frame.e_AP, expected.e_AP)) < -0.99


def test_build_femoral_frame_returns_none_with_fewer_than_two_landmarks() -> None:
    mesh = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert build_femoral_frame([], mesh) is None
    assert build_femoral_frame([{"id": "intercondylar_notch", "physical": [0.0, 0.0, 0.0]}], mesh) is None


def test_build_femoral_frame_marks_lm_degenerate_low_confidence() -> None:
    mesh = _box_mesh()
    landmarks = [
        {"id": "lateral_condylar_edge", "physical": [0.0, 0.0, 0.0]},
        {"id": "medial_condylar_edge", "physical": [0.0, 0.0, 0.001]},
        {"id": "intercondylar_groove_midpoint", "physical": [0.0, -1.0, 0.0]},
        {"id": "intercondylar_notch", "physical": [0.0, 1.0, 0.0]},
    ]

    frame = build_femoral_frame(landmarks, mesh)

    assert frame is not None
    assert frame.confidence == "low"
    assert "ml_pca_fallback_edges_too_close" in frame.evidence["flags"]


def test_build_femoral_frame_marks_coplanar_landmarks_low_confidence() -> None:
    mesh = _box_mesh()
    landmarks = [
        {"id": "lateral_condylar_edge", "physical": [0.0, 0.0, 2.0]},
        {"id": "medial_condylar_edge", "physical": [0.0, 0.0, -2.0]},
        {"id": "intercondylar_groove_midpoint", "physical": [0.0, 0.0, 0.0]},
        {"id": "intercondylar_notch", "physical": [0.0, 0.0, 2.0]},
    ]

    frame = build_femoral_frame(landmarks, mesh)

    assert frame is not None
    assert frame.confidence == "low"
    assert "pca_fallback_coplanar_landmarks" in frame.evidence["flags"]



def test_build_femoral_frame_g_only_is_medium_confidence() -> None:
    landmarks = [
        {"id": "lateral_condylar_edge", "physical": [0.0, 0.0, 2.0]},
        {"id": "medial_condylar_edge", "physical": [0.0, 0.0, -2.0]},
        {"id": "intercondylar_groove_midpoint", "physical": [0.0, -1.0, 0.0]},
    ]

    frame = build_femoral_frame(landmarks, _box_mesh())

    assert frame is not None
    assert frame.confidence == "medium"


def test_build_femoral_frame_n_only_is_medium_confidence() -> None:
    landmarks = [
        {"id": "lateral_condylar_edge", "physical": [0.0, 0.0, 2.0]},
        {"id": "medial_condylar_edge", "physical": [0.0, 0.0, -2.0]},
        {"id": "intercondylar_notch", "physical": [0.0, 1.0, 0.0]},
    ]

    frame = build_femoral_frame(landmarks, _box_mesh())

    assert frame is not None
    assert frame.confidence == "medium"


def test_recession_differential_confirms_when_density_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("microct_analysis.processing.femoral_frame._DEFAULT_DENSITY_RATIO_THRESHOLD", 1_000_000.0)
    monkeypatch.setattr("microct_analysis.processing.femoral_frame.local_ap_recession", lambda point, *_args, **_kwargs: float(point[1]))

    frame = build_femoral_frame(_recession_landmarks(), _box_mesh())

    assert frame is not None
    assert frame.evidence["ap_verification"] == "recession_differential_confirmed"
    assert np.allclose(frame.e_AP, [0.0, -1.0, 0.0])


def test_recession_differential_flips_swapped_when_density_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("microct_analysis.processing.femoral_frame._DEFAULT_DENSITY_RATIO_THRESHOLD", 1_000_000.0)
    monkeypatch.setattr("microct_analysis.processing.femoral_frame.local_ap_recession", lambda point, *_args, **_kwargs: float(point[1]))
    landmarks = _recession_landmarks()
    landmarks[2]["physical"], landmarks[3]["physical"] = landmarks[3]["physical"], landmarks[2]["physical"]

    frame = build_femoral_frame(landmarks, _box_mesh())

    assert frame is not None
    assert frame.evidence["ap_verification"] == "recession_differential_flipped"
    assert frame.confidence == "medium"


def test_condylar_region_mask_selects_peak_ml_span_region() -> None:
    mesh = np.vstack([
        np.column_stack([np.full(5, -5.0), np.zeros(5), np.linspace(-0.5, 0.5, 5)]),
        np.column_stack([np.zeros(9), np.zeros(9), np.linspace(-3.0, 3.0, 9)]),
        np.column_stack([np.full(5, 5.0), np.zeros(5), np.linspace(-0.5, 0.5, 5)]),
    ])

    selected = mesh[condylar_region_mask(mesh)]

    assert selected[:, 0].min() <= 0.0 <= selected[:, 0].max()
    assert np.ptp(selected[:, 2]) >= 6.0

def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def _box_mesh() -> np.ndarray:
    z, y, x = np.meshgrid(np.linspace(-2.0, 2.0, 5), np.linspace(-2.0, 2.0, 5), np.linspace(-2.0, 2.0, 5))
    return np.column_stack([z.ravel(), y.ravel(), x.ravel()])


def _recession_landmarks() -> list[dict]:
    return [
        {"id": "lateral_condylar_edge", "physical": [0.0, 0.0, 2.0]},
        {"id": "medial_condylar_edge", "physical": [0.0, 0.0, -2.0]},
        {"id": "intercondylar_groove_midpoint", "physical": [0.0, -1.0, 0.0]},
        {"id": "intercondylar_notch", "physical": [0.0, 1.0, 0.0]},
    ]
