"""Landmark-derived femoral anatomical frame construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from microct_analysis.processing.surface import condylar_region_mask, local_ap_recession

_COORDINATE_COUNT = 3
_MIN_ML_WIDTH_MM = 1.0
_MIN_AP_SEPARATION_MM = 1.0
_MIN_PERP_NORM_MM = 0.01
# Calibrated on real OA6-1RK condylar mesh: correct frame density ratio
# is 9.11 with synthesized verified condylar-edge landmarks, while swapped
# G/N is 0.11. The design default 1.3 cleanly separates both cases, so it
# remains intentionally conservative rather than OA6-specific.
_DEFAULT_DENSITY_RATIO_THRESHOLD = 1.3
# OA6-1RK produced a 1,421,885-vertex midline band at 30% condylar width;
# keep the design default because it leaves ample density-check support.
_DEFAULT_MIDLINE_WIDTH_FRACTION = 0.30
_MIN_MIDLINE_VERTICES = 20

_LATERAL_ID = "lateral_condylar_edge"
_MEDIAL_ID = "medial_condylar_edge"
_GROOVE_ID = "intercondylar_groove_midpoint"
_NOTCH_ID = "intercondylar_notch"


@dataclass(frozen=True)
class FemoralFrame:
    e_ML: np.ndarray
    e_AP: np.ndarray
    e_SI: np.ndarray
    confidence: str
    evidence: dict[str, Any]


def build_femoral_frame(placed_landmarks: list[dict], mesh_vertices: np.ndarray) -> FemoralFrame | None:
    """Build a right-handed ZYX femoral frame from placed landmarks and mesh geometry."""

    points = _as_vertices(mesh_vertices)
    landmarks = _landmark_points(placed_landmarks)
    if len(landmarks) < 2:
        return None

    evidence: dict[str, Any] = {"warnings": [], "flags": []}
    confidence = "high"

    lateral = landmarks.get(_LATERAL_ID)
    medial = landmarks.get(_MEDIAL_ID)
    groove = landmarks.get(_GROOVE_ID)
    notch = landmarks.get(_NOTCH_ID)

    condylar_mask = condylar_region_mask(points)
    condylar = points[condylar_mask]
    if len(condylar) < _MIN_MIDLINE_VERTICES:
        condylar = points
        condylar_mask = np.ones(len(points), dtype=bool)
        evidence["warnings"].append("condylar_region_too_small")
        confidence = _lower_confidence(confidence, "medium")
    evidence["condylar_vertex_count"] = int(len(condylar))

    if lateral is None or medial is None:
        e_ml = _pca_axis(condylar, axis_index=2)
        confidence = "low"
        evidence["flags"].append("ml_pca_fallback_missing_edge")
        center = np.mean(condylar, axis=0)
    else:
        center = (lateral + medial) / 2.0
        ml_raw = lateral - medial
        ml_norm = float(np.linalg.norm(ml_raw))
        evidence["condylar_width_mm"] = ml_norm
        if ml_norm < _MIN_ML_WIDTH_MM:
            e_ml = _pca_axis(condylar, axis_index=2)
            confidence = "low"
            evidence["flags"].append("ml_pca_fallback_edges_too_close")
        else:
            e_ml = ml_raw / ml_norm

    d, d_source, source_confidence = _ap_seed_vector(groove, notch, center, points)
    confidence = _lower_confidence(confidence, source_confidence)
    evidence["ap_seed_source"] = d_source

    d_perp = d - float(np.dot(d, e_ml)) * e_ml
    d_perp_norm = float(np.linalg.norm(d_perp))
    evidence["ap_seed_perpendicular_norm_mm"] = d_perp_norm
    ap_sign_reference = d_perp.copy() if d_perp_norm >= _MIN_PERP_NORM_MM else None
    if d_perp_norm < _MIN_PERP_NORM_MM:
        e_si = _pca_axis(points, axis_index=0)
        e_ap = np.cross(e_si, e_ml)
        ap_norm = float(np.linalg.norm(e_ap))
        if ap_norm < _MIN_PERP_NORM_MM:
            e_ap = _orthogonal_pca_axis(points, e_ml)
            e_si = np.cross(e_ml, e_ap)
        else:
            e_ap = e_ap / ap_norm
        e_si = _unit(np.cross(e_ml, e_ap))
        confidence = "low"
        evidence["flags"].append("pca_fallback_coplanar_landmarks")
    else:
        d_perp = d_perp / d_perp_norm
        e_si = _unit(np.cross(e_ml, d_perp))
        e_ap = _unit(np.cross(e_si, e_ml))
        if float(np.dot(e_ap, d_perp)) > 0:
            e_ap = -e_ap
            e_si = -e_si

    e_ap, e_si, ap_evidence, confidence = _verify_ap_sign(
        e_ml=e_ml,
        e_ap=e_ap,
        e_si=e_si,
        center=center,
        condylar=condylar,
        groove=groove,
        notch=notch,
        confidence=confidence,
    )
    evidence.update(ap_evidence)

    centroid_all = np.mean(points, axis=0)
    si_centroid_projection = float(np.dot(e_si, centroid_all - center))
    evidence["si_centroid_projection_mm"] = si_centroid_projection
    if si_centroid_projection <= 0:
        e_si = -e_si
        e_ap = -e_ap
        evidence["flags"].append("si_flipped_toward_shaft_bulk")

    if ap_sign_reference is not None and float(np.dot(e_ap, ap_sign_reference)) > 0:
        e_ap = -e_ap
        e_si = -e_si
        evidence["flags"].append("ap_sign_restored_anterior_positive")

    e_ml = _unit(e_ml)
    e_ap = _unit(e_ap - float(np.dot(e_ap, e_ml)) * e_ml)
    e_si = _unit(np.cross(e_ml, e_ap))
    _assert_orthonormal(e_ml, e_ap, e_si)

    evidence["midline_band_vertex_count"] = int(evidence.get("midline_band_vertex_count", 0))
    evidence["confidence"] = confidence
    return FemoralFrame(e_ML=e_ml, e_AP=e_ap, e_SI=e_si, confidence=confidence, evidence=evidence)



def serialize_derived_frame(frame: FemoralFrame, source_landmark_ids: list[str]) -> dict[str, Any]:
    """Serialize a derived femoral frame for positions.json."""

    return {
        "ml_vector": frame.e_ML.tolist(),
        "ap_vector": frame.e_AP.tolist(),
        "si_vector": frame.e_SI.tolist(),
        "confidence": frame.confidence,
        "source_landmarks": source_landmark_ids,
        "ap_verification": frame.evidence.get("ap_verification"),
    }

def _verify_ap_sign(
    *,
    e_ml: np.ndarray,
    e_ap: np.ndarray,
    e_si: np.ndarray,
    center: np.ndarray,
    condylar: np.ndarray,
    groove: np.ndarray | None,
    notch: np.ndarray | None,
    confidence: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    evidence: dict[str, Any] = {}
    width_reference = max(_axis_span(condylar, e_ml), _MIN_ML_WIDTH_MM)
    epsilon = _DEFAULT_MIDLINE_WIDTH_FRACTION * width_reference
    ml_projection = np.abs((condylar - center) @ e_ml)
    midline = condylar[ml_projection < epsilon]
    evidence["midline_band_half_width_mm"] = float(epsilon)
    evidence["midline_band_vertex_count"] = int(len(midline))
    evidence["density_ratio_threshold"] = _DEFAULT_DENSITY_RATIO_THRESHOLD

    if groove is not None and notch is not None:
        tree = KDTree(condylar)
        groove_recession = local_ap_recession(groove, condylar, tree)
        notch_recession = local_ap_recession(notch, condylar, tree)
        evidence.update(
            {
                "groove_recession_mm": float(groove_recession),
                "notch_recession_mm": float(notch_recession),
                "recession_differential_mm": float(notch_recession - groove_recession),
            }
        )

    if len(midline) >= _MIN_MIDLINE_VERTICES:
        ap_projection = (midline - center) @ e_ap
        anterior = int(np.count_nonzero(ap_projection > 0))
        posterior = int(np.count_nonzero(ap_projection < 0))
        ratio = float(anterior / max(posterior, 1))
        evidence.update(
            {
                "ap_density_anterior_count": anterior,
                "ap_density_posterior_count": posterior,
                "ap_density_ratio": ratio,
            }
        )
        if ratio > _DEFAULT_DENSITY_RATIO_THRESHOLD:
            evidence["ap_verification"] = "mesh_density_confirmed"
            return e_ap, e_si, evidence, confidence
        if ratio < 1.0 / _DEFAULT_DENSITY_RATIO_THRESHOLD:
            evidence["ap_verification"] = "mesh_density_flipped"
            confidence = _lower_confidence(confidence, "medium")
            return -e_ap, -e_si, evidence, confidence
        evidence["ap_verification"] = "mesh_density_inconclusive"
    else:
        evidence["ap_verification"] = "mesh_density_skipped"

    if groove is None or notch is None:
        evidence["recession_differential_mm"] = None
        return e_ap, e_si, evidence, confidence

    differential = float(evidence["recession_differential_mm"])
    if differential > 0:
        evidence["ap_verification"] = "recession_differential_confirmed"
        return e_ap, e_si, evidence, confidence

    evidence["ap_verification"] = "recession_differential_flipped"
    confidence = _lower_confidence(confidence, "medium")
    return -e_ap, -e_si, evidence, confidence


def _landmark_points(placed_landmarks: list[dict]) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for landmark in placed_landmarks:
        landmark_id = landmark.get("id")
        if not landmark_id or "physical" not in landmark:
            continue
        point = np.asarray(landmark["physical"], dtype=float)
        if point.shape == (_COORDINATE_COUNT,):
            points[str(landmark_id)] = point
    return points


def _ap_seed_vector(
    groove: np.ndarray | None,
    notch: np.ndarray | None,
    center: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    if groove is not None and notch is not None:
        d = notch - groove
        if float(np.linalg.norm(d)) >= _MIN_AP_SEPARATION_MM:
            return d, "groove_to_notch", "high"
        return _mesh_centroid_ap_fallback(center, vertices), "mesh_centroid_ap_fallback_close_gn", "medium"
    if groove is not None:
        return center - groove, "groove_to_centroid", "medium"
    if notch is not None:
        return notch - center, "centroid_to_notch", "medium"
    return _mesh_centroid_ap_fallback(center, vertices), "mesh_centroid_ap_fallback_missing_gn", "low"


def _mesh_centroid_ap_fallback(center: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    centroid = np.mean(vertices, axis=0)
    d = centroid - center
    if float(np.linalg.norm(d)) == 0:
        d = _pca_axis(vertices, axis_index=1)
    return d


def _pca_axis(vertices: np.ndarray, *, axis_index: int) -> np.ndarray:
    centered = vertices - np.mean(vertices, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[min(axis_index, vh.shape[0] - 1)]
    return _unit(axis)


def _orthogonal_pca_axis(vertices: np.ndarray, e_ml: np.ndarray) -> np.ndarray:
    centered = vertices - np.mean(vertices, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    for axis in vh:
        candidate = axis - float(np.dot(axis, e_ml)) * e_ml
        norm = float(np.linalg.norm(candidate))
        if norm > _MIN_PERP_NORM_MM:
            return candidate / norm
    raise ValueError("could not derive an axis orthogonal to ML")


def _axis_span(vertices: np.ndarray, axis: np.ndarray) -> float:
    projected = vertices @ axis
    return float(np.ptp(projected))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("cannot normalize zero-length vector")
    return np.asarray(vector, dtype=float) / norm


def _lower_confidence(current: str, candidate: str) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    return candidate if order[candidate] > order[current] else current


def _assert_orthonormal(e_ml: np.ndarray, e_ap: np.ndarray, e_si: np.ndarray) -> None:
    assert abs(float(np.dot(e_ml, e_ap))) < 1e-10
    assert abs(float(np.dot(e_ml, e_si))) < 1e-10
    assert abs(float(np.dot(e_ap, e_si))) < 1e-10
    basis = np.stack([e_ml, e_ap, e_si], axis=0)
    assert abs(float(np.linalg.det(basis)) - 1.0) < 1e-8


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != _COORDINATE_COUNT:
        raise ValueError("mesh_vertices must have shape (n, 3) in (Z, Y, X) order")
    if len(points) == 0:
        raise ValueError("mesh_vertices must not be empty")
    return points
