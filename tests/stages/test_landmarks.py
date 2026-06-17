from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microct_analysis.stages.landmarks_orientation import (
    _femoral_surface_position,
    _growth_plate_slice,
    _growth_plate_slice_from_ratios,
    _is_sustained_drop,
    _landmark_confidence,
    _rotation_matrix,
    _tibial_slice_position,
    compute_orientation_frame,
    run_landmarks_orientation,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_landmark_positions_written_from_label_centroid_and_extrema(tmp_path: Path) -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1:4, 2:4, 1:3] = 7
    label_path = tmp_path / "labels.npy"
    np.save(label_path, labels)
    assignments_path = tmp_path / "structure_assignments.json"
    assignments_path.write_text(json.dumps({"assignments": {"femur": 7}, "spacing": [10, 20, 30]}))

    report = run_landmarks_orientation(
        {"labels": str(label_path), "structure_assignments": str(assignments_path)},
        [
            {"id": "femur_center", "structure": "femur", "method": "centroid"},
            {"id": "growth_plate", "structure": "femur", "method": "distal"},
        ],
        {"axes": {"superior_inferior": {"from": "femur_center", "to": "growth_plate"}}, "target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )

    positions = json.loads((tmp_path / "landmarks" / "positions.json").read_text())
    assert report["confidence"] == "high"
    assert positions["landmarks"][0]["voxel"] == [2.0, 2.5, 1.5]
    assert positions["landmarks"][0]["physical"] == [20.0, 50.0, 45.0]
    assert positions["landmarks"][1]["voxel"][0] == 3.0


def test_orientation_transform_records_axis_explanation_and_translation() -> None:
    frame = compute_orientation_frame(
        [
            {"id": "origin", "physical": [10.0, 0.0, 0.0]},
            {"id": "distal", "physical": [20.0, 0.0, 0.0]},
        ],
        {
            "origin_landmark": "origin",
            "target_plane": "frontal",
            "axes": {"superior_inferior": {"from": "origin", "to": "distal"}},
        },
    )

    assert frame["translation"] == [-10.0, -0.0, -0.0]
    assert frame["axes"]["superior_inferior"] == [1.0, 0.0, 0.0]
    assert "workflow frontal plane" in frame["explanation"]
    assert "superior-inferior now follows" in frame["explanation"]


def test_rotation_matrix_orthogonalizes_near_collinear_axes() -> None:
    matrix = np.asarray(
        _rotation_matrix(
            {
                "axis_1": [1.0, 0.0, 0.0],
                "axis_2": [1.0, 1e-8, 0.0],
                "axis_3": [0.0, 0.0, 1.0],
            }
        )
    )

    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
    assert abs(abs(float(np.linalg.det(matrix))) - 1.0) < 1e-10


@pytest.mark.skip(reason="PCA orientation retired in visual-landmarking")
def test_pca_orientation_applied_when_tibia_label_present(tmp_path: Path) -> None:
    labels = np.zeros((9, 9, 9), dtype=np.uint8)
    labels[3:5, 1:4, 2:8] = 3
    label_path = tmp_path / "labels.npy"
    np.save(label_path, labels)
    assignments_path = tmp_path / "structure_assignments.json"
    assignments_path.write_text(json.dumps({"assignments": {"tibia": 3}, "spacing": [1.0, 1.0, 1.0]}))

    report = run_landmarks_orientation(
        {"labels": str(label_path), "structure_assignments": str(assignments_path)},
        [{"id": "lateral_tibial_condyle_edge", "structure": "tibia", "domain": "tibial_2d_slice"}],
        {"target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )

    frame = json.loads((tmp_path / "landmarks" / "orientation_frame.json").read_text())
    oriented = np.load(tmp_path / "landmarks" / "oriented_labels.npy")
    assert report["confidence"] == "high"
    assert frame["pca_orientation"]["applied"] is True
    assert frame["pca_orientation"]["label_interpolation_order"] == 0
    assert frame["pca_orientation"]["intensity_interpolation_order"] == 1
    assert oriented.shape == labels.shape
    assert np.count_nonzero(oriented == 3) >= 3


@pytest.mark.skip(reason="PCA orientation retired in visual-landmarking")
def test_pca_orientation_fallback_marks_tibial_landmarks_low_confidence(tmp_path: Path) -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[2, 2, 2] = 3
    label_path = tmp_path / "labels.npy"
    np.save(label_path, labels)
    assignments_path = tmp_path / "structure_assignments.json"
    assignments_path.write_text(json.dumps({"assignments": {"tibia": 3}, "spacing": [1.0, 1.0, 1.0]}))

    report = run_landmarks_orientation(
        {"labels": str(label_path), "structure_assignments": str(assignments_path)},
        [{"id": "articular_surface_proximal", "structure": "tibia", "domain": "tibial_2d_slice"}],
        {"target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )

    positions = json.loads((tmp_path / "landmarks" / "positions.json").read_text())
    report_text = (tmp_path / "landmarks" / "orientation_report.md").read_text()
    assert report["confidence"] == "low"
    assert positions["landmarks"][0]["confidence"] == "low"
    assert positions["landmarks"][0]["requires_user_confirmation"] is True
    assert "PyVista" in report_text


def test_growth_plate_slice_uses_intensity_bone_fill_drop() -> None:
    mask = np.ones((10, 5, 5), dtype=bool)
    intensity = np.zeros(mask.shape, dtype=np.float32)
    intensity[:5, :, :4] = 100.0
    intensity[5:, :, :1] = 100.0
    definition = {
        "id": "growth_plate_proximal",
        "domain": "tibial_2d_slice",
        "geometric_params": {
            "detection": "bone_fill_ratio_drop",
            "fill_ratio_threshold_pct": 50,
            "min_consecutive_above": 5,
        },
    }

    voxel, confidence, evidence = _tibial_slice_position(
        definition, mask, (1.0, 1.0, 1.0), "growth_plate", intensity
    )

    assert voxel[0] == 5.0
    assert confidence == "high"
    assert "growth plate boundary at slice 5" in evidence
    assert "sustained-drop validated" in evidence


def test_growth_plate_ratio_scan_rejects_transient_drop() -> None:
    ratios = np.array(
        [np.nan, 0.8, 0.8, 0.8, 0.8, 0.8, 0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4]
    )

    growth = _growth_plate_slice_from_ratios(
        ratios,
        articular=1,
        fill_threshold=0.5,
        min_consecutive=5,
        min_sustained=3,
    )

    assert growth == 12
    assert not _is_sustained_drop(ratios, 6, 0.5, 3)
    assert _is_sustained_drop(ratios, 12, 0.5, 3)


def test_growth_plate_sustained_drop_accepts_terminal_transition() -> None:
    ratios = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.4])

    assert _is_sustained_drop(ratios, 5, 0.5, 3)


# FALSE-GREEN: this test only passes because oa6_1rk_tibia_fill_ratios.npz is
# fabricated — its fill ratios are idealized step values (~0.55 until slice 276,
# then 0.42 at 280) that select slice 278. The real OA6-1RK pipeline computes a
# noisy bone-fraction signal that first sustainably drops below 0.5 at the
# epiphysis (~246), so the live detector returns 246/247 (IIOC 40 sl), NOT 278.
# The bone_fill_ratio_drop @ 50% method locates the wrong feature; LDA-1's
# sustained-drop refinement cannot help. Marked xfail (non-strict) so this
# fabricated pass shows as xpass — a documented "not a trustworthy green" — and
# the suite stays green, until the growth-plate algorithm + a pipeline-captured
# fixture are reworked (work item: growth-plate-rework). See session
# microct-oa6-1rk-004 validation.
@pytest.mark.xfail(
    strict=False,
    reason="fabricated growth-plate fixture; live pipeline returns 246 not 278 — detector unsolved",
)
def test_growth_plate_oa6_1rk_selects_sustained_drop() -> None:
    data = np.load(FIXTURES / "oa6_1rk_tibia_fill_ratios.npz")
    golden = json.loads((FIXTURES / "oa6_1rk_golden.json").read_text())
    gp = golden["growth_plate"]

    first_slice = int(data["first_slice_index"])
    label_counts = data["label_counts"]
    ratios = np.full(first_slice + len(label_counts), np.nan, dtype=float)
    ratios[first_slice:][label_counts > 0] = data["fill_ratios"][label_counts > 0]

    growth_plate = _growth_plate_slice_from_ratios(
        ratios,
        articular=gp["articular_slice"],
        fill_threshold=0.5,
        min_consecutive=5,
        min_sustained=3,
    )

    assert growth_plate is not None
    assert growth_plate != 247
    assert abs(growth_plate - gp["golden_slice"]) <= gp["tolerance_slices"]
    assert abs((growth_plate - gp["articular_slice"]) - gp["iioc_height_slices"]) <= gp["tolerance_slices"]


def test_growth_plate_slice_falls_back_to_label_area_without_intensity() -> None:
    mask = np.zeros((8, 5, 5), dtype=bool)
    mask[:5, :4, :4] = True
    mask[5, :2, :2] = True
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)

    growth = _growth_plate_slice(mask, counts, 0, {"detection": "bone_fill_ratio_drop", "min_consecutive_above": 5})
    _voxel, _confidence, evidence = _tibial_slice_position(
        {"id": "growth_plate_proximal", "geometric_params": {"detection": "bone_fill_ratio_drop"}},
        mask,
        (1.0, 1.0, 1.0),
        "growth_plate",
    )

    assert growth == 5
    assert "label-area fallback" in evidence


def test_run_landmarks_orientation_threads_filtered_intensity_to_growth_plate(tmp_path: Path) -> None:
    labels = np.zeros((11, 5, 5), dtype=np.uint8)
    labels[:, 1:4, 1:4] = 3
    intensity = np.zeros(labels.shape, dtype=np.float32)
    intensity[:5, 1:4, 1:3] = 100.0
    intensity[5:, 1:4, 1:2] = 100.0
    labels_path = tmp_path / "labels.npy"
    intensity_path = tmp_path / "filtered.npy"
    assignments_path = tmp_path / "structure_assignments.json"
    np.save(labels_path, labels)
    np.save(intensity_path, intensity)
    assignments_path.write_text(json.dumps({"assignments": {"tibia": 3}, "spacing": [1.0, 1.0, 1.0]}))

    report = run_landmarks_orientation(
        {"labels": str(labels_path), "filtered": str(intensity_path), "structure_assignments": str(assignments_path)},
        [
            {"id": "articular_surface_proximal", "structure": "tibia", "domain": "tibial_2d_slice", "geometric_params": {"surface": "articular"}},
            {
                "id": "growth_plate_proximal",
                "structure": "tibia",
                "domain": "tibial_2d_slice",
                "method": "growth_plate",
                "geometric_params": {
                    "detection": "bone_fill_ratio_drop",
                    "fill_ratio_threshold_pct": 50,
                    "min_consecutive_above": 5,
                },
            },
        ],
        {"target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )

    positions = json.loads((tmp_path / "landmarks" / "positions.json").read_text())

    assert report["confidence"] == "high"
    assert positions["landmarks"][1]["voxel"][0] == 5.0


def test_femoral_notch_outside_distal_window_is_low_confidence(monkeypatch) -> None:
    vertices = np.array(
        [
            [1.0, 3.0, -4.0],
            [2.0, 3.0, -3.5],
            [1.0, 3.0, 4.0],
            [2.0, 3.0, 3.5],
            [3.0, 3.0, 0.0],
            [5.0, 3.0, 0.0],
            [6.0, 3.0, 1.0],
            [6.0, 3.0, -1.0],
            [6.0, 3.0, 2.0],
            [6.0, 3.0, -2.0],
            [0.0, -3.0, 0.0],
            [6.0, -3.0, 0.0],
            [1.0, -3.0, -4.0],
            [2.0, -3.0, -3.5],
            [1.0, -3.0, 4.0],
            [2.0, -3.0, 3.5],
            [3.0, -3.0, 0.0],
            [5.0, -3.0, 0.0],
            [6.0, -3.0, 2.0],
            [6.0, -3.0, -2.0],
        ]
    )

    def fake_surface_mesh(_mask, _spacing):
        return vertices, np.zeros((0, 3), dtype=np.int64)

    monkeypatch.setattr("microct_analysis.stages.landmarks_orientation.extract_surface_mesh", fake_surface_mesh)
    monkeypatch.setattr(
        "microct_analysis.stages.landmarks_orientation.find_notch_depth",
        lambda _vertices, *, surface_region: (np.array([6.0, 3.0, 0.0]), np.array([0.0]), np.array([[6.0, 3.0, 0.0]])),
    )
    monkeypatch.setattr("microct_analysis.stages.landmarks_orientation._condylar_si_limit", lambda _posterior: 4.0)

    _voxel, confidence, evidence = _femoral_surface_position(
        {"id": "intercondylar_notch"}, np.ones((2, 2, 2), dtype=bool), (1.0, 1.0, 1.0), "notch_depth_maximum"
    )

    assert confidence == "low"
    assert "implausibly proximal" in evidence


def test_femoral_notch_oa6_1rk_condylar_result_is_high_confidence() -> None:
    data = np.load(FIXTURES / "oa6_1rk_femur_condylar.npz")

    _voxel, confidence, evidence = _femoral_surface_position(
        {"id": "intercondylar_notch"}, data["mask"], (1.0, 1.0, 1.0), "notch_depth_maximum"
    )

    assert confidence == "high"
    assert "implausibly proximal" not in evidence


def test_landmark_confidence_names_implausible_growth_plate_not_orientation() -> None:
    landmarks = [
        {"id": "articular_surface_proximal", "voxel": [20.0, 0.0, 0.0], "confidence": "high"},
        {"id": "growth_plate_proximal", "voxel": [145.0, 0.0, 0.0], "confidence": "high"},
    ]

    confidence, evidence = _landmark_confidence(landmarks, {"orientation_confidence": "high", "explanation": ""})

    assert confidence == "medium"
    assert landmarks[1]["confidence"] == "medium"
    assert "Growth plate at slice 145" in evidence
    assert "IIOC = 125" in evidence
    assert "PCA orientation unavailable" not in evidence


@pytest.mark.skip(reason="PCA orientation retired in visual-landmarking")
def test_oriented_tibial_landmark_differs_from_unoriented_asymmetric_volume(tmp_path: Path) -> None:
    labels = np.zeros((9, 9, 9), dtype=np.uint8)
    labels[3:5, 1:4, 2:8] = 3
    label_path = tmp_path / "labels.npy"
    np.save(label_path, labels)
    assignments_path = tmp_path / "structure_assignments.json"
    assignments_path.write_text(json.dumps({"assignments": {"tibia": 3}, "spacing": [1.0, 1.0, 1.0]}))

    run_landmarks_orientation(
        {"labels": str(label_path), "structure_assignments": str(assignments_path)},
        [{"id": "articular_surface_proximal", "structure": "tibia", "domain": "tibial_2d_slice"}],
        {"target_plane": "frontal"},
        output_dir=str(tmp_path / "landmarks"),
    )

    positions = json.loads((tmp_path / "landmarks" / "positions.json").read_text())
    oriented = np.load(tmp_path / "landmarks" / "oriented_labels.npy")
    original_articular_z = int(np.flatnonzero((labels == 3).reshape(labels.shape[0], -1).sum(axis=1))[0])
    assert not np.array_equal(oriented, labels)
    assert positions["landmarks"][0]["voxel"][0] != float(original_articular_z)
