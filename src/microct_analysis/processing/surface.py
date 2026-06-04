"""Surface mesh extraction and anatomical landmark heuristics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from skimage.measure import marching_cubes

_COORDINATE_COUNT = 3
_SI_AXIS = 0
_AP_AXIS = 1
_ML_AXIS = 2


def extract_surface_mesh(label_mask: np.ndarray, spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes on a binary label mask.

    Vertices are returned in physical millimetre coordinates using the same
    ``(Z, Y, X)`` / ``(SI, AP, ML)`` axis order as the input volume.
    """

    mask = np.asarray(label_mask, dtype=bool)
    if mask.ndim != _COORDINATE_COUNT:
        raise ValueError("label_mask must be a 3D array")
    if len(spacing) != _COORDINATE_COUNT:
        raise ValueError("spacing must contain three values in (z, y, x) order")
    if any(axis_spacing <= 0 for axis_spacing in spacing):
        raise ValueError("spacing values must be positive")
    if not mask.any():
        raise ValueError("label_mask must contain foreground voxels")
    if mask.all():
        raise ValueError("label_mask must contain background voxels for surface extraction")

    vertices, faces, _normals, _values = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    return vertices, faces


def find_saddle_point(vertices: np.ndarray, *, surface_region: str = "anterior_distal") -> np.ndarray:
    """Find the femoral intercondylar groove midpoint on the anterior-distal surface."""

    points = _as_vertices(vertices)
    if surface_region != "anterior_distal":
        raise ValueError("only surface_region='anterior_distal' is supported")

    anterior = points[points[:, _AP_AXIS] < np.median(points[:, _AP_AXIS])]
    if anterior.size == 0:
        raise ValueError("could not identify anterior surface vertices")

    distal = anterior[anterior[:, _SI_AXIS] < np.median(anterior[:, _SI_AXIS])]
    if distal.size == 0:
        raise ValueError("could not identify anterior-distal surface vertices")

    tree = KDTree(distal)
    neighbor_count = min(12, len(distal))
    _distances, neighbor_indices = tree.query(distal, k=neighbor_count)
    if neighbor_count == 1:
        local_ml_curvature = np.zeros(len(distal), dtype=float)
    else:
        neighbor_ml = distal[neighbor_indices, _ML_AXIS]
        local_ml_curvature = np.abs(distal[:, _ML_AXIS] - np.mean(neighbor_ml, axis=1))

    ml_midline = np.median(distal[:, _ML_AXIS])
    midline_distance = np.abs(distal[:, _ML_AXIS] - ml_midline)
    superior_bonus = _normalize(distal[:, _SI_AXIS])

    score = _normalize(midline_distance) + _normalize(local_ml_curvature) - (2.0 * superior_bonus)
    return distal[int(np.argmin(score))].copy()


def find_notch_depth(vertices: np.ndarray, *, surface_region: str = "posterior_intercondylar") -> np.ndarray:
    """Find the deepest posterior intercondylar notch point."""

    if surface_region != "posterior_intercondylar":
        raise ValueError("only surface_region='posterior_intercondylar' is supported")
    points = _as_vertices(vertices)
    posterior = points[points[:, _AP_AXIS] > np.median(points[:, _AP_AXIS])]
    if posterior.size == 0:
        raise ValueError("could not identify posterior surface vertices")

    si_cutoff = np.median(posterior[:, _SI_AXIS])
    distal_posterior = posterior[posterior[:, _SI_AXIS] <= si_cutoff]
    if distal_posterior.size == 0:
        raise ValueError("could not identify distal posterior surface vertices")

    ml_midline = np.median(points[:, _ML_AXIS])
    ml_span = np.ptp(points[:, _ML_AXIS])
    tolerance = max(ml_span * 0.1, np.finfo(float).eps)
    midline = distal_posterior[np.abs(distal_posterior[:, _ML_AXIS] - ml_midline) <= tolerance]
    if midline.size == 0:
        distances = np.abs(distal_posterior[:, _ML_AXIS] - ml_midline)
        keep_count = max(1, min(len(distal_posterior), len(points) // 20))
        midline = distal_posterior[np.argsort(distances)[:keep_count]]

    return midline[int(np.argmax(midline[:, _SI_AXIS]))].copy()


def find_condylar_edge(
    vertices: np.ndarray,
    direction: str,
    *,
    include_osteophytes: bool = True,
) -> np.ndarray:
    """Find the outermost medial or lateral condylar surface point."""

    points = _as_vertices(vertices)
    if direction not in {"medial", "lateral"}:
        raise ValueError("direction must be 'medial' or 'lateral'")
    if not include_osteophytes:
        raise NotImplementedError("exclude-osteophyte condylar edge detection is not implemented")

    distal = points[points[:, _SI_AXIS] < np.median(points[:, _SI_AXIS])]
    if distal.size == 0:
        raise ValueError("could not identify distal surface vertices")

    edge_index = np.argmax(distal[:, _ML_AXIS]) if direction == "lateral" else np.argmin(distal[:, _ML_AXIS])
    return distal[int(edge_index)].copy()


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != _COORDINATE_COUNT:
        raise ValueError("vertices must have shape (n, 3) in (Z, Y, X) order")
    if len(points) == 0:
        raise ValueError("vertices must not be empty")
    return points


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    if span == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.min(values)) / span
