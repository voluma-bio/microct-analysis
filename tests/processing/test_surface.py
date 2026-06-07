from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microct_analysis.processing.surface import (
    extract_surface_mesh,
    find_condylar_edge,
    find_notch_depth,
    find_saddle_point,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_extract_surface_mesh_produces_vertices_and_faces_with_expected_shapes() -> None:
    z, y, x = np.indices((24, 28, 32))
    mask = ((z - 12) / 7) ** 2 + ((y - 14) / 8) ** 2 + ((x - 16) / 10) ** 2 <= 1

    vertices, faces = extract_surface_mesh(mask, spacing=(0.2, 0.3, 0.4))

    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert np.all(vertices >= 0)
    assert vertices[:, 0].max() > vertices[:, 0].min()


def test_find_condylar_edge_returns_extreme_ml_points_from_distal_surface() -> None:
    vertices = _dumbbell_vertices()

    medial = find_condylar_edge(vertices, "medial")
    lateral = find_condylar_edge(vertices, "lateral")

    distal = vertices[vertices[:, 0] < np.median(vertices[:, 0])]
    assert medial[2] == np.min(distal[:, 2])
    assert lateral[2] == np.max(distal[:, 2])


def test_find_saddle_point_returns_anterior_distal_midline_groove_vertex() -> None:
    vertices = _dumbbell_vertices()
    expected = np.array([2.5, -2.7, 0.0])

    saddle, scores, verts = find_saddle_point(vertices)

    assert np.allclose(saddle, expected)

    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0
    assert np.array_equal(saddle, verts[int(np.argmin(scores))])


def test_find_notch_depth_returns_distal_posterior_midline_notch_not_shaft() -> None:
    vertices = _notch_fixture_vertices()

    notch, scores, verts = find_notch_depth(vertices)

    assert np.allclose(notch, [4.0, 2.5, 0.0])
    assert notch[0] < 5.0
    assert notch[2] == 0.0

    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0
    assert np.array_equal(notch, verts[int(np.argmin(scores))])


def test_notch_oa6_1rk_finds_intercondylar_concavity() -> None:
    data = np.load(FIXTURES / "oa6_1rk_femur_condylar.npz")
    golden = json.loads((FIXTURES / "oa6_1rk_golden.json").read_text())

    vertices, _faces = extract_surface_mesh(data["mask"], spacing=(1.0, 1.0, 1.0))
    vertices[:, 0] += int(data["z_offset"])
    vertices[:, 1] += int(data["y_offset"])
    vertices[:, 2] += int(data["x_offset"])

    notch, _scores, _verts = find_notch_depth(vertices)
    groove = np.array(golden["notch"]["groove_position"])
    spacing = np.array(golden["spacing_mm"])
    dfl = float(np.sqrt(np.sum(((notch - groove) * spacing) ** 2)))
    dfl_lo, dfl_hi = golden["notch"]["dfl_range_mm"]

    assert dfl_lo <= dfl <= dfl_hi
    assert notch[0] > groove[0] + golden["notch"]["min_notch_groove_si_gap"]


def _dumbbell_vertices() -> np.ndarray:
    left_condyle = np.array(
        [
            [1.0, -2.0, -5.0],
            [2.0, -2.5, -4.5],
            [3.0, 2.0, -5.2],
            [4.0, 1.5, -3.8],
        ]
    )
    right_condyle = np.array(
        [
            [1.0, -2.0, 5.0],
            [2.0, -2.5, 4.5],
            [3.0, 2.0, 5.2],
            [4.0, 1.5, 3.8],
        ]
    )
    groove = np.array(
        [
            [4.8, -3.0, 0.0],
            [3.5, -2.8, -0.8],
            [3.5, -2.8, 0.8],
            [2.5, -2.7, 0.0],
        ]
    )
    proximal_noise = np.array(
        [
            [8.0, -3.2, 0.0],
            [9.0, 0.0, -1.0],
            [9.0, 0.0, 1.0],
        ]
    )
    return np.vstack([left_condyle, right_condyle, groove, proximal_noise])


def _notch_fixture_vertices() -> np.ndarray:
    condylar_surface = np.array(
        [
            [1.0, 4.0, -4.0],
            [1.0, 4.0, -3.5],
            [1.0, 4.0, 3.5],
            [1.0, 4.0, 4.0],
            [2.0, 4.0, -4.0],
            [2.0, 4.0, -3.5],
            [2.0, 4.0, 3.5],
            [2.0, 4.0, 4.0],
            [3.0, 4.0, -4.0],
            [3.0, 4.0, -3.5],
            [3.0, 4.0, 3.5],
            [3.0, 4.0, 4.0],
            [4.0, 4.0, -4.0],
            [4.0, 4.0, -3.5],
            [4.0, 4.0, 3.5],
            [4.0, 4.0, 4.0],
        ]
    )
    notch = np.array([[4.0, 2.5, 0.0]])
    shaft = np.array(
        [
            [6.0, 3.0, -1.0],
            [6.0, 3.0, 0.0],
            [6.0, 3.0, 1.0],
            [8.0, 3.0, 0.0],
            [7.0, 3.0, -1.0],
            [8.0, 3.0, 1.0],
            [9.0, 3.0, 0.0],
        ]
    )
    anterior = np.array(
        [[float(z), -3.0, float(x)] for z in range(10) for x in (-4, -2, 0, 2)]
    )
    return np.vstack([condylar_surface, notch, shaft, anterior])
