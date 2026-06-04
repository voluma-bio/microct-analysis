from __future__ import annotations

import numpy as np

from microct_analysis.processing.surface import (
    extract_surface_mesh,
    find_condylar_edge,
    find_notch_depth,
    find_saddle_point,
)


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

    saddle = find_saddle_point(vertices)

    assert np.allclose(saddle, expected)


def test_find_notch_depth_returns_distal_posterior_midline_notch_not_shaft() -> None:
    vertices = _notch_fixture_vertices()

    notch = find_notch_depth(vertices)

    assert np.allclose(notch, [3.0, 3.0, 0.0])
    assert notch[0] < 5.0
    assert notch[2] == 0.0


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
    left_posterior = np.array(
        [
            [1.0, 3.0, -4.0],
            [2.0, 3.0, -3.5],
        ]
    )
    right_posterior = np.array(
        [
            [1.0, 3.0, 4.0],
            [2.0, 3.0, 3.5],
        ]
    )
    notch = np.array([[3.0, 3.0, 0.0]])
    shaft = np.array(
        [
            [8.0, 3.0, 0.0],
            [7.0, 3.0, -1.0],
            [8.0, 3.0, 1.0],
            [9.0, 3.0, 0.0],
        ]
    )
    anterior = np.array(
        [
            [2.0, -3.0, 0.0],
            [5.0, -2.0, 0.0],
            [1.0, -3.0, -4.0],
            [2.0, -3.0, -3.5],
            [1.0, -3.0, 4.0],
            [2.0, -3.0, 3.5],
            [7.0, -3.0, -1.0],
            [8.0, -3.0, 1.0],
            [9.0, -3.0, 0.0],
        ]
    )
    return np.vstack([left_posterior, right_posterior, notch, shaft, anterior])
