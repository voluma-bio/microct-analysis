"""Mesh cleanup helpers for landmarking."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_COORDINATE_COUNT = 3
_FACE_VERTEX_COUNT = 3


def preprocess_femoral_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the largest face-connected mesh component by vertex count.

    Vertices are expected in ZYX physical coordinates. If ``faces`` is None,
    component membership is unavailable; the vertices are returned unchanged
    and the returned faces value is None.
    """

    points = np.asarray(vertices)
    if points.ndim != 2 or points.shape[1] != _COORDINATE_COUNT:
        raise ValueError("vertices must have shape (n, 3) in (Z, Y, X) order")
    if len(points) == 0:
        raise ValueError("vertices must not be empty")
    if faces is None:
        return points.copy(), None

    face_array = np.asarray(faces)
    if face_array.ndim != 2 or face_array.shape[1] != _FACE_VERTEX_COUNT:
        raise ValueError("faces must have shape (n, 3)")
    if len(face_array) == 0:
        return points.copy(), face_array.copy()
    if face_array.min() < 0 or face_array.max() >= len(points):
        raise ValueError("faces contain vertex indices outside vertices")

    edges = np.vstack(
        [
            face_array[:, [0, 1]],
            face_array[:, [1, 2]],
            face_array[:, [2, 0]],
        ]
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(points), len(points)))
    _component_count, labels = connected_components(graph, directed=False, return_labels=True)

    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    keep_label = int(np.argmax(counts))
    keep_vertices = labels == keep_label
    old_to_new = np.full(len(points), -1, dtype=np.int64)
    old_to_new[keep_vertices] = np.arange(int(np.count_nonzero(keep_vertices)))

    keep_faces = np.all(keep_vertices[face_array], axis=1)
    cleaned_faces = old_to_new[face_array[keep_faces]].astype(face_array.dtype, copy=False)
    return points[keep_vertices].copy(), cleaned_faces.copy()
