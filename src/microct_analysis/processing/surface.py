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
    """Find the posterior intercondylar notch roof via scored concavity."""

    if surface_region != "posterior_intercondylar":
        raise ValueError("only surface_region='posterior_intercondylar' is supported")
    points = _as_vertices(vertices)
    posterior = points[points[:, _AP_AXIS] > np.median(points[:, _AP_AXIS])]
    if posterior.size == 0:
        raise ValueError("could not identify posterior surface vertices")

    si_limit = _condylar_si_limit(posterior)
    condylar = posterior[posterior[:, _SI_AXIS] <= si_limit]
    if len(condylar) < 4:
        raise ValueError("could not identify condylar posterior vertices")

    tree = KDTree(condylar)
    neighbor_count = min(12, len(condylar))
    _distances, neighbor_indices = tree.query(condylar, k=neighbor_count)
    if neighbor_count == 1:
        local_ml_curvature = np.zeros(len(condylar), dtype=float)
        local_ap_recession = np.zeros(len(condylar), dtype=float)
    else:
        neighbor_ml = condylar[neighbor_indices, _ML_AXIS]
        local_ml_curvature = np.abs(condylar[:, _ML_AXIS] - np.mean(neighbor_ml, axis=1))
        neighbor_ap = condylar[neighbor_indices, _AP_AXIS]
        local_ap_recession = np.mean(neighbor_ap, axis=1) - condylar[:, _AP_AXIS]

    ml_midline = np.median(points[:, _ML_AXIS])
    midline_distance = np.abs(condylar[:, _ML_AXIS] - ml_midline)
    ap_recession = local_ap_recession + _local_ap_recession(condylar)
    superior_bonus = _normalize(condylar[:, _SI_AXIS])

    score = (
        _normalize(midline_distance)
        + _normalize(local_ml_curvature)
        - (1.5 * _normalize(ap_recession))
        - (2.0 * superior_bonus)
    )
    return condylar[int(np.argmin(score))].copy()


def _local_ap_recession(condylar: np.ndarray) -> np.ndarray:
    si = condylar[:, _SI_AXIS]
    n_bins = max(10, min(50, len(condylar) // 500))
    bin_edges = np.linspace(si.min(), si.max(), n_bins + 1)
    recession = np.zeros(len(condylar), dtype=float)
    for index in range(n_bins):
        upper = si <= bin_edges[index + 1] if index == n_bins - 1 else si < bin_edges[index + 1]
        in_bin = (si >= bin_edges[index]) & upper
        if np.any(in_bin):
            recession[in_bin] = np.percentile(condylar[in_bin, _AP_AXIS], 95) - condylar[in_bin, _AP_AXIS]
    return recession


def _condylar_si_limit(posterior: np.ndarray, *, ml_span_fraction: float = 0.5) -> float:
    """Estimate the SI boundary between condylar region and shaft."""

    si = posterior[:, _SI_AXIS]
    n_bins = max(10, min(50, len(posterior) // 500))
    bin_edges = np.linspace(si.min(), si.max(), n_bins + 1)

    ml_spans = np.zeros(n_bins)
    valid_bins = np.zeros(n_bins, dtype=bool)
    for index in range(n_bins):
        upper = si <= bin_edges[index + 1] if index == n_bins - 1 else si < bin_edges[index + 1]
        in_bin = posterior[(si >= bin_edges[index]) & upper]
        if len(in_bin) >= 3:
            ml_spans[index] = np.ptp(in_bin[:, _ML_AXIS])
            valid_bins[index] = True

    peak_span = ml_spans.max()
    if peak_span == 0:
        return float(si.max())

    peak_bin = int(np.argmax(ml_spans))
    threshold = peak_span * ml_span_fraction
    for index in range(peak_bin + 1, n_bins):
        if valid_bins[index] and ml_spans[index] < threshold:
            return float(bin_edges[index])
    return float(si.max())


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
