"""Per-landmark quantitative validation (backstop) signals.

The backstop does NOT find landmarks — the agent does. The backstop VALIDATES
a candidate placement and either accepts, rejects-with-feedback, or flags-HITL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from skimage.filters import threshold_otsu

from microct_analysis.processing.femoral_frame import FemoralFrame
from microct_analysis.processing.surface import condylar_region_mask, local_ap_recession, local_ml_curvature


@dataclass(frozen=True)
class BackstopResult:
    """Result of backstop validation for a single landmark."""

    accepted: bool
    confidence: str  # "high", "medium", "low"
    signals: dict[str, dict]  # per-signal details {name: {value, accept, note}}
    feedback: str  # human-readable feedback for retry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_backstop(
    landmark_def: dict[str, Any],
    coordinate: tuple[float, float, float],
    mesh_vertices: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    placed_landmarks: dict[str, Any] | None = None,
    femoral_frame: FemoralFrame | None = None,
) -> BackstopResult:
    """Dispatch to per-landmark-type backstop signals, apply ensemble rule.

    Dispatches by domain:
    - femoral_3d_surface -> surface backstop. Coordinates and femoral
      placed landmarks are physical millimetres from the surface mesh.
    - tibial_2d_slice -> slice backstop. Coordinates and cross-landmark
      intervals are voxel/slice indices (IIOC remains slice-count based).
    - unknown -> generic backstop (bone membership only)
    """
    domain = str(landmark_def.get("domain", ""))
    landmark_id = str(landmark_def.get("id") or landmark_def.get("name", ""))

    if domain == "femoral_3d_surface":
        return _femoral_backstop(
            landmark_id, coordinate, mesh_vertices, spacing, placed_landmarks, femoral_frame
        )
    elif domain == "tibial_2d_slice":
        return _tibial_backstop(
            landmark_id, coordinate, labels, intensity, spacing, placed_landmarks
        )
    else:
        return _generic_backstop(coordinate, labels, spacing)


# ---------------------------------------------------------------------------
# Femoral surface backstop
# ---------------------------------------------------------------------------

_SI_AXIS = 0
_AP_AXIS = 1
_ML_AXIS = 2
_K_NEIGHBORS = 12


def _femoral_backstop(
    landmark_id: str,
    coordinate: tuple[float, float, float],
    mesh_vertices: np.ndarray | None,
    spacing: tuple[float, float, float],
    placed_landmarks: dict[str, Any] | None,
    femoral_frame: FemoralFrame | None = None,
) -> BackstopResult:
    """Femoral surface landmark backstop with frame-free Pass 1 and optional frame-dependent Pass 2."""
    signals: dict[str, dict] = {}

    if mesh_vertices is None or len(mesh_vertices) == 0:
        return BackstopResult(accepted=True, confidence="medium", signals={}, feedback="")

    coord = np.asarray(coordinate, dtype=float)
    verts = np.asarray(mesh_vertices, dtype=float)
    tree = KDTree(verts)

    dist, _ = tree.query(coord)
    snap_tolerance = 1.0
    signals["bone_membership"] = {
        "value": float(dist),
        "accept": bool(dist <= snap_tolerance),
        "note": f"snap distance {dist:.3f} (tolerance {snap_tolerance})",
    }

    if landmark_id in ("lateral_condylar_edge", "medial_condylar_edge"):
        signals.update(_condylar_edge_signals(coord, verts, tree, landmark_id))
    elif landmark_id == "intercondylar_groove_midpoint":
        signals.update(_groove_signals(coord, verts, tree))
    elif landmark_id == "intercondylar_notch":
        signals.update(_notch_signals(coord, verts, tree))

    if placed_landmarks is not None:
        signals.update(_cross_landmark_check(landmark_id, coordinate, placed_landmarks, spacing))

    if femoral_frame is not None and placed_landmarks is not None:
        signals.update(_frame_dependent_signals(landmark_id, coord, verts, tree, placed_landmarks, femoral_frame))

    return _femoral_ensemble_decision(signals)


def _groove_signals(coord: np.ndarray, verts: np.ndarray, tree: KDTree) -> dict[str, dict]:
    """Frame-free groove checks: ML midline proximity and local saddle curvature."""
    signals: dict[str, dict] = {}
    signals.update(_ml_midline_signal(coord, verts))
    signals.update(_condylar_region_signal(coord, verts))
    curvature = local_ml_curvature(coord, verts, tree, k=_K_NEIGHBORS)
    signals["local_saddle_curvature"] = {
        "value": float(curvature),
        "accept": bool(curvature >= 0.0),
        "note": f"local ML curvature {curvature:.6f}",
    }
    return signals


def _notch_signals(coord: np.ndarray, verts: np.ndarray, tree: KDTree) -> dict[str, dict]:
    """Frame-free notch checks: ML midline proximity and AP recession."""
    signals = _ml_midline_signal(coord, verts)
    signals.update(_condylar_region_signal(coord, verts))
    recession = local_ap_recession(coord, verts, tree, k=_K_NEIGHBORS)
    signals["local_ap_recession"] = {
        "value": float(recession),
        "accept": bool(recession > 0),
        "note": f"local AP recession {recession:.6f} (positive is recessed)",
    }
    return signals


def _ml_midline_signal(coord: np.ndarray, verts: np.ndarray) -> dict[str, dict]:
    ml_midline = float(np.median(verts[:, _ML_AXIS]))
    ml_dist = abs(float(coord[_ML_AXIS]) - ml_midline)
    ml_range = float(np.ptp(verts[:, _ML_AXIS]))
    tolerance = ml_range * 0.3 if ml_range > 0 else 1.0
    return {
        "ml_midline_proximity": {
            "value": float(ml_dist),
            "accept": bool(ml_dist <= tolerance),
            "note": f"ML distance from midline {ml_dist:.3f} (tolerance {tolerance:.3f})",
        }
    }



def _condylar_region_signal(coord: np.ndarray, verts: np.ndarray) -> dict[str, dict]:
    mask = condylar_region_mask(verts)
    if not np.any(mask):
        return {}
    _dist, nearest_index = KDTree(verts).query(coord)
    accepted = bool(mask[int(nearest_index)])
    return {
        "si_in_condylar_region": {
            "value": accepted,
            "accept": accepted,
            "note": "candidate nearest surface vertex is in mesh-level condylar region",
        }
    }

def _condylar_edge_signals(
    coord: np.ndarray,
    verts: np.ndarray,
    tree: KDTree,
    landmark_id: str,
) -> dict[str, dict]:
    """Frame-free condylar edge checks using a condylar-region-restricted ML extremum."""
    signals: dict[str, dict] = {}
    condylar_mask = condylar_region_mask(verts)
    condylar = verts[condylar_mask]
    if len(condylar) == 0:
        condylar = verts
        condylar_mask = np.ones(len(verts), dtype=bool)

    tree_all = KDTree(verts)
    _dist, nearest_index = tree_all.query(coord)
    in_condylar_region = bool(condylar_mask[int(nearest_index)])
    ml_range = float(np.ptp(condylar[:, _ML_AXIS]))
    if "lateral" in landmark_id:
        extreme = float(np.max(condylar[:, _ML_AXIS]))
        threshold = extreme - ml_range * 0.2
        accepted = bool(in_condylar_region and coord[_ML_AXIS] >= threshold)
        note = (
            f"ML={coord[_ML_AXIS]:.2f} vs lateral threshold={threshold:.2f} in condylar region; "
            f"SI in condylar region={in_condylar_region}"
        )
    else:
        extreme = float(np.min(condylar[:, _ML_AXIS]))
        threshold = extreme + ml_range * 0.2
        accepted = bool(in_condylar_region and coord[_ML_AXIS] <= threshold)
        note = (
            f"ML={coord[_ML_AXIS]:.2f} vs medial threshold={threshold:.2f} in condylar region; "
            f"SI in condylar region={in_condylar_region}"
        )
    signals["ml_extremity"] = {"value": float(coord[_ML_AXIS]), "accept": accepted, "note": note}

    _distances, neighbor_indices = tree.query(coord, k=min(_K_NEIGHBORS, len(verts)))
    neighbors = verts[np.atleast_1d(neighbor_indices)]
    neighbor_mean_ml = float(np.mean(neighbors[:, _ML_AXIS]))
    ml_delta = float(coord[_ML_AXIS] - neighbor_mean_ml)
    convex = ml_delta > 0 if "lateral" in landmark_id else ml_delta < 0
    signals["local_convexity"] = {
        "value": ml_delta,
        "accept": bool(convex),
        "note": "candidate is outward from local neighbor mean in ML direction",
    }
    return signals


def _frame_dependent_signals(
    landmark_id: str,
    coord: np.ndarray,
    verts: np.ndarray,
    tree: KDTree,
    placed_landmarks: dict[str, Any],
    frame: FemoralFrame,
) -> dict[str, dict]:
    signals: dict[str, dict] = {}
    groove = _get_placed_coordinate(placed_landmarks, "intercondylar_groove_midpoint")
    notch = _get_placed_coordinate(placed_landmarks, "intercondylar_notch")
    lateral = _get_placed_coordinate(placed_landmarks, "lateral_condylar_edge")
    medial = _get_placed_coordinate(placed_landmarks, "medial_condylar_edge")

    if landmark_id == "intercondylar_groove_midpoint":
        groove = coord
    elif landmark_id == "intercondylar_notch":
        notch = coord
    elif landmark_id == "lateral_condylar_edge":
        lateral = coord
    elif landmark_id == "medial_condylar_edge":
        medial = coord

    if groove is not None and notch is not None and landmark_id in ("intercondylar_groove_midpoint", "intercondylar_notch"):
        ap_delta = float(np.dot(notch - groove, frame.e_AP))
        signals["notch_posterior_to_groove"] = {
            "value": ap_delta,
            "accept": bool(ap_delta < 0),
            "note": f"dot(N-G, e_AP)={ap_delta:.6f}; notch must be posterior to groove",
        }
        groove_recession = local_ap_recession(groove, verts, tree, k=_K_NEIGHBORS)
        notch_recession = local_ap_recession(notch, verts, tree, k=_K_NEIGHBORS)
        diff = float(notch_recession - groove_recession)
        signals["recession_differential"] = {
            "value": diff,
            "accept": bool(diff > 0),
            "note": f"notch recession - groove recession = {diff:.6f}",
        }
        si_delta = float(np.dot(notch - groove, frame.e_SI))
        signals["notch_groove_si_ordering"] = {
            "value": si_delta,
            "accept": bool(si_delta >= -0.05),
            "note": f"dot(N-G, e_SI)={si_delta:.6f}; notch must be more proximal",
        }

    if lateral is not None and medial is not None and landmark_id in ("lateral_condylar_edge", "medial_condylar_edge"):
        si_delta_abs = abs(float(np.dot(lateral - medial, frame.e_SI)))
        si_range = _axis_range(verts, frame.e_SI)
        tolerance = 0.2 * si_range
        signals["condylar_edges_si_parity"] = {
            "value": si_delta_abs,
            "accept": bool(si_delta_abs < tolerance),
            "note": f"SI delta {si_delta_abs:.6f} (tolerance {tolerance:.6f})",
        }
    return signals


def _axis_range(verts: np.ndarray, axis: np.ndarray) -> float:
    return float(np.ptp(verts @ np.asarray(axis, dtype=float)))


# ---------------------------------------------------------------------------
# Tibial slice backstop
# ---------------------------------------------------------------------------


def _tibial_backstop(
    landmark_id: str,
    coordinate: tuple[float, float, float],
    labels: np.ndarray | None,
    intensity: np.ndarray | None,
    spacing: tuple[float, float, float],
    placed_landmarks: dict[str, Any] | None,
) -> BackstopResult:
    """Tibial slice landmark backstop.

    Signals for articular_surface_proximal:
    - area_threshold_crossing: first slice >= 20% max area

    Signals for growth_plate_proximal (CRITICAL - 4 signals, 2 families):
    - bone_fill_ratio_drop (intensity family)
    - intensity_gradient_magnitude (intensity family)
    - texture_homogeneity_drop (intensity family)
    - mask_area_gradient (geometry family - independent of intensity!)

    Ensemble rule for growth plate: MAJORITY VOTE requiring agreement
    from >=2 signal FAMILIES (not just >=2 signals).

    Signals for condyle edges:
    - slice_bone_extent: edge is at ML extreme
    - tibial_width: cross-landmark width plausibility

    Hard veto (all tibial):
    - iioc_plausibility: growth_plate - articular must be 50-100 slices
    """
    signals: dict[str, dict] = {}
    coord = np.asarray(coordinate, dtype=float)
    candidate_slice = int(round(coord[0]))

    # --- Generic bone membership ---
    if labels is not None:
        z, y, x = (
            int(round(coord[0])),
            int(round(coord[1])),
            int(round(coord[2])),
        )
        if (
            0 <= z < labels.shape[0]
            and 0 <= y < labels.shape[1]
            and 0 <= x < labels.shape[2]
        ):
            on_bone = int(labels[z, y, x]) > 0
            signals["bone_membership"] = {
                "value": on_bone,
                "accept": bool(on_bone),
                "note": "on bone" if on_bone else "off bone",
            }
        else:
            signals["bone_membership"] = {
                "value": False,
                "accept": False,
                "note": "coordinate out of volume bounds",
            }

    # --- Per-landmark signals ---
    if landmark_id == "articular_surface_proximal":
        if labels is not None:
            signals.update(_articular_signals(candidate_slice, labels))
    elif landmark_id == "growth_plate_proximal":
        if labels is not None and intensity is not None:
            gp_signals = _growth_plate_signals(
                candidate_slice, labels, intensity
            )
            signals.update(gp_signals)
    elif landmark_id in ("lateral_condylar_edge", "medial_condylar_edge"):
        if labels is not None:
            signals.update(
                _tibial_edge_signals(coord, labels, landmark_id)
            )

    # --- Cross-landmark signals (IIOC veto) ---
    if placed_landmarks is not None:
        cross = _cross_landmark_check(
            landmark_id, coordinate, placed_landmarks, spacing
        )
        signals.update(cross)

    # --- Growth plate uses family-majority vote ---
    if landmark_id == "growth_plate_proximal":
        return _growth_plate_ensemble(signals)

    return _ensemble_decision(signals)


def _articular_signals(candidate_slice: int, labels: np.ndarray) -> dict[str, dict]:
    """Area threshold crossing: first slice >= 20% max area."""
    mask = labels > 0
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float)
    max_area = areas.max()
    if max_area == 0:
        return {}
    threshold = max_area * 0.2
    # Find first slice crossing threshold
    crossing_slice = None
    for z in range(mask.shape[0]):
        if areas[z] >= threshold:
            crossing_slice = z
            break
    if crossing_slice is None:
        return {}
    # The candidate should be within 10 slices of the crossing
    distance = abs(candidate_slice - crossing_slice)
    signals: dict[str, dict] = {}
    signals["area_threshold_crossing"] = {
        "value": float(distance),
        "accept": bool(distance <= 10),
        "note": f"candidate slice {candidate_slice} vs area crossing slice {crossing_slice} (distance {distance})",
    }
    return signals


def _growth_plate_signals(
    candidate_slice: int, labels: np.ndarray, intensity: np.ndarray
) -> dict[str, dict]:
    """Growth plate signals: 4 signals from 2 families."""
    signals: dict[str, dict] = {}
    mask = labels > 0

    # --- Intensity family ---
    # 1. bone_fill_ratio_drop
    bone_threshold = _derive_bone_threshold(mask, intensity)
    fill_ratios = _compute_fill_ratios(mask, intensity, bone_threshold)
    valid_ratios = ~np.isnan(fill_ratios)
    if valid_ratios.sum() > 5:
        median_fill = float(np.nanmedian(fill_ratios))
        drop_threshold = median_fill * 0.5
        # Find first slice where fill ratio drops below half the median
        fill_drop_slice = None
        for z in range(mask.shape[0]):
            if not valid_ratios[z]:
                continue
            if fill_ratios[z] < drop_threshold:
                # Check sustained
                below_count = 0
                for zz in range(z, min(z + 5, mask.shape[0])):
                    if valid_ratios[zz] and fill_ratios[zz] < drop_threshold:
                        below_count += 1
                if below_count >= 3:
                    fill_drop_slice = z
                    break
        if fill_drop_slice is not None:
            distance = abs(candidate_slice - fill_drop_slice)
            signals["bone_fill_ratio_drop"] = {
                "value": float(distance),
                "accept": bool(distance <= 15),
                "family": "intensity",
                "note": f"fill ratio drop at slice {fill_drop_slice}, candidate at {candidate_slice} (distance {distance})",
            }
        else:
            signals["bone_fill_ratio_drop"] = {
                "value": None,
                "accept": True,  # no signal = don't penalize
                "family": "intensity",
                "note": "no clear fill ratio drop detected",
            }

    # 2. intensity_gradient_magnitude
    grad_slice = _intensity_gradient_signal(mask, intensity)
    if grad_slice is not None:
        distance = abs(candidate_slice - grad_slice)
        signals["intensity_gradient_magnitude"] = {
            "value": float(distance),
            "accept": bool(distance <= 15),
            "family": "intensity",
            "note": f"intensity gradient spike at slice {grad_slice}, candidate at {candidate_slice} (distance {distance})",
        }
    else:
        signals["intensity_gradient_magnitude"] = {
            "value": None,
            "accept": True,
            "family": "intensity",
            "note": "no clear intensity gradient spike detected",
        }

    # 3. texture_homogeneity_drop
    texture_slice = _texture_homogeneity_signal(mask, intensity)
    if texture_slice is not None:
        distance = abs(candidate_slice - texture_slice)
        signals["texture_homogeneity_drop"] = {
            "value": float(distance),
            "accept": bool(distance <= 15),
            "family": "intensity",
            "note": f"texture drop at slice {texture_slice}, candidate at {candidate_slice} (distance {distance})",
        }
    else:
        signals["texture_homogeneity_drop"] = {
            "value": None,
            "accept": True,
            "family": "intensity",
            "note": "no clear texture homogeneity drop detected",
        }

    # --- Geometry family ---
    # 4. mask_area_gradient
    area_slice = _mask_area_gradient_signal(mask)
    if area_slice is not None:
        distance = abs(candidate_slice - area_slice)
        signals["mask_area_gradient"] = {
            "value": float(distance),
            "accept": bool(distance <= 15),
            "family": "geometry",
            "note": f"area gradient spike at slice {area_slice}, candidate at {candidate_slice} (distance {distance})",
        }
    else:
        signals["mask_area_gradient"] = {
            "value": None,
            "accept": True,
            "family": "geometry",
            "note": "no clear mask area gradient spike detected",
        }

    return signals


def _tibial_edge_signals(
    coord: np.ndarray, labels: np.ndarray, landmark_id: str
) -> dict[str, dict]:
    """Tibial condylar edge: check ML extreme position within slice."""
    signals: dict[str, dict] = {}
    z = int(round(coord[0]))
    if z < 0 or z >= labels.shape[0]:
        return signals
    slice_mask = labels[z] > 0
    if not slice_mask.any():
        return signals
    # Find ML extent in the slice
    ys, xs = np.where(slice_mask)
    if len(xs) == 0:
        return signals
    if "lateral" in landmark_id:
        extreme_x = float(np.max(xs))
        at_extreme = coord[2] >= extreme_x * 0.7
        signals["slice_bone_extent"] = {
            "value": float(coord[2]),
            "accept": bool(at_extreme),
            "note": f"X={coord[2]:.1f} vs lateral extreme={extreme_x:.1f}",
        }
    else:
        extreme_x = float(np.min(xs))
        ml_range = float(np.ptp(xs))
        threshold = extreme_x + ml_range * 0.3
        at_extreme = coord[2] <= threshold
        signals["slice_bone_extent"] = {
            "value": float(coord[2]),
            "accept": bool(at_extreme),
            "note": f"X={coord[2]:.1f} vs medial threshold={threshold:.1f}",
        }
    return signals


# ---------------------------------------------------------------------------
# Growth plate family-majority ensemble
# ---------------------------------------------------------------------------

_GROWTH_PLATE_SIGNAL_IDS = (
    "bone_fill_ratio_drop",
    "intensity_gradient_magnitude",
    "texture_homogeneity_drop",
    "mask_area_gradient",
)


def _growth_plate_ensemble(signals: dict[str, dict]) -> BackstopResult:
    """Majority vote for growth plate requiring >=2 signal FAMILIES to agree.

    Intensity family signals: bone_fill_ratio_drop, intensity_gradient_magnitude,
                             texture_homogeneity_drop
    Geometry family signals: mask_area_gradient
    """
    # Separate growth plate signals from others
    gp_signals = {
        k: v for k, v in signals.items() if k in _GROWTH_PLATE_SIGNAL_IDS
    }
    other_signals = {
        k: v for k, v in signals.items() if k not in _GROWTH_PLATE_SIGNAL_IDS
    }

    # Check hard vetoes first (IIOC, bone membership)
    hard_reject = any(
        not s.get("accept", True)
        for k, s in other_signals.items()
        if k == "iioc_plausibility"
    )
    if hard_reject:
        feedback_parts = [
            s["note"]
            for s in other_signals.values()
            if not s.get("accept", True)
        ]
        return BackstopResult(
            accepted=False,
            confidence="low",
            signals=signals,
            feedback="; ".join(feedback_parts),
        )

    # Family vote: each family votes accept if ANY signal in that family accepts
    intensity_signals = [
        v for k, v in gp_signals.items() if v.get("family") == "intensity"
    ]
    geometry_signals = [
        v for k, v in gp_signals.items() if v.get("family") == "geometry"
    ]

    # A family accepts if majority of its signals with definite values accept
    def family_accepts(family_signals: list[dict]) -> bool | None:
        """Return True if family accepts, False if rejects, None if no signal."""
        if not family_signals:
            return None
        accepting = sum(1 for s in family_signals if s.get("accept", True))
        return accepting > len(family_signals) / 2

    intensity_vote = family_accepts(intensity_signals)
    geometry_vote = family_accepts(geometry_signals)

    # Count accepting families
    family_votes = [v for v in (intensity_vote, geometry_vote) if v is not None]
    accepting_families = sum(1 for v in family_votes if v)

    if len(family_votes) == 0:
        # No growth plate signals available
        return _ensemble_decision(other_signals)

    if accepting_families >= min(2, len(family_votes)):
        # Both families (or the only available family) accept
        confidence = "high" if accepting_families == len(family_votes) else "medium"
        return BackstopResult(
            accepted=True,
            confidence=confidence,
            signals=signals,
            feedback="",
        )
    else:
        # Insufficient family agreement
        rejecting = [
            s["note"] for s in gp_signals.values() if not s.get("accept", True)
        ]
        return BackstopResult(
            accepted=False,
            confidence="low",
            signals=signals,
            feedback="; ".join(rejecting) if rejecting else "insufficient family agreement",
        )


# ---------------------------------------------------------------------------
# Cross-landmark validation
# ---------------------------------------------------------------------------


def _cross_landmark_check(
    landmark_id: str,
    coordinate: tuple[float, float, float],
    placed_landmarks: dict[str, Any],
    spacing: tuple[float, float, float],
) -> dict[str, dict]:
    """Check consistency between placed landmarks.

    Femoral surface landmarks are already physical millimetres and are not
    multiplied by spacing. Tibial IIOC landmarks remain voxel/slice indices.
    """
    signals: dict[str, dict] = {}
    coord = np.asarray(coordinate, dtype=float)

    # DFL range (groove-notch distance)
    if landmark_id in ("intercondylar_groove_midpoint", "intercondylar_notch"):
        other_id = (
            "intercondylar_notch"
            if "groove" in landmark_id
            else "intercondylar_groove_midpoint"
        )
        other = _get_placed_coordinate(placed_landmarks, other_id)
        if other is not None:
            dfl = float(np.linalg.norm(coord - other))
            signals["dfl_range"] = {
                "value": dfl,
                "accept": bool(1.5 <= dfl <= 3.1),
                "note": f"DFL = {dfl:.3f}mm (range 1.5-3.1mm)",
            }

    # Condylar width
    if landmark_id in ("lateral_condylar_edge", "medial_condylar_edge"):
        other_id = (
            "medial_condylar_edge"
            if "lateral" in landmark_id
            else "lateral_condylar_edge"
        )
        other = _get_placed_coordinate(placed_landmarks, other_id)
        if other is not None:
            width = float(np.linalg.norm(coord - other))
            signals["condylar_width"] = {
                "value": width,
                "accept": bool(2.0 <= width <= 5.0),
                "note": f"Condylar width = {width:.3f}mm (range 2.0-5.0mm)",
            }

    # IIOC plausibility
    if landmark_id in ("articular_surface_proximal", "growth_plate_proximal"):
        other_id = (
            "growth_plate_proximal"
            if "articular" in landmark_id
            else "articular_surface_proximal"
        )
        other = _get_placed_coordinate(placed_landmarks, other_id)
        if other is not None:
            iioc_slices = abs(coord[0] - other[0])
            signals["iioc_plausibility"] = {
                "value": float(iioc_slices),
                "accept": bool(50 <= iioc_slices <= 100),
                "note": f"IIOC = {iioc_slices:.0f} slices (range 50-100)",
            }

    return signals


def _get_placed_coordinate(
    placed_landmarks: dict[str, Any], landmark_id: str
) -> np.ndarray | None:
    """Extract coordinate from placed_landmarks dict."""
    if placed_landmarks is None:
        return None
    for item in placed_landmarks.get("landmarks", []):
        if isinstance(item, dict) and item.get("id") == landmark_id:
            coord = item.get("physical") or item.get("voxel")
            if coord is not None:
                return np.asarray(coord, dtype=float)
    # Also check direct key access
    if landmark_id in placed_landmarks:
        return np.asarray(placed_landmarks[landmark_id], dtype=float)
    return None


# ---------------------------------------------------------------------------
# Generic backstop
# ---------------------------------------------------------------------------


def _generic_backstop(
    coordinate: tuple[float, float, float],
    labels: np.ndarray | None,
    spacing: tuple[float, float, float],
) -> BackstopResult:
    """Generic backstop for unknown landmark domains. Checks bone membership only."""
    signals: dict[str, dict] = {}
    if labels is not None:
        z, y, x = (
            int(round(coordinate[0])),
            int(round(coordinate[1])),
            int(round(coordinate[2])),
        )
        if (
            0 <= z < labels.shape[0]
            and 0 <= y < labels.shape[1]
            and 0 <= x < labels.shape[2]
        ):
            on_bone = int(labels[z, y, x]) > 0
            signals["bone_membership"] = {
                "value": on_bone,
                "accept": bool(on_bone),
                "note": "on bone" if on_bone else "off bone",
            }
        else:
            signals["bone_membership"] = {
                "value": False,
                "accept": False,
                "note": "coordinate out of volume bounds",
            }

    all_accept = all(s.get("accept", True) for s in signals.values())
    any_reject = any(not s.get("accept", True) for s in signals.values())

    if all_accept:
        return BackstopResult(
            accepted=True, confidence="high", signals=signals, feedback=""
        )
    elif any_reject:
        feedback_parts = [
            s["note"] for s in signals.values() if not s.get("accept", True)
        ]
        return BackstopResult(
            accepted=False,
            confidence="low",
            signals=signals,
            feedback="; ".join(feedback_parts),
        )
    else:
        return BackstopResult(
            accepted=True, confidence="medium", signals=signals, feedback=""
        )



def _femoral_ensemble_decision(signals: dict[str, dict]) -> BackstopResult:
    """Femoral ensemble with bone membership as a hard surface-landmark veto."""

    bone = signals.get("bone_membership")
    if bone is not None and not bone.get("accept", True):
        return BackstopResult(
            accepted=False,
            confidence="low",
            signals=signals,
            feedback=str(bone.get("note", "candidate is off mesh")),
        )
    return _ensemble_decision(signals)

# ---------------------------------------------------------------------------
# Ensemble decision (standard — non-growth-plate)
# ---------------------------------------------------------------------------


def _ensemble_decision(signals: dict[str, dict]) -> BackstopResult:
    """Standard ensemble: all accept=high, any hard reject=low, mixed=medium."""
    if not signals:
        return BackstopResult(
            accepted=True, confidence="medium", signals=signals, feedback=""
        )

    # Hard vetoes
    hard_veto_keys = {
        "iioc_plausibility",
        "ml_extremity",
        "si_in_condylar_region",
        "dfl_range",
        "notch_posterior_to_groove",
        "recession_differential",
        "notch_groove_si_ordering",
    }
    hard_reject = any(
        not s.get("accept", True)
        for k, s in signals.items()
        if k in hard_veto_keys
    )
    if hard_reject:
        feedback_parts = [
            s["note"] for s in signals.values() if not s.get("accept", True)
        ]
        return BackstopResult(
            accepted=False,
            confidence="low",
            signals=signals,
            feedback="; ".join(feedback_parts),
        )

    all_accept = all(s.get("accept", True) for s in signals.values())
    any_reject = any(not s.get("accept", True) for s in signals.values())

    if all_accept:
        return BackstopResult(
            accepted=True, confidence="high", signals=signals, feedback=""
        )
    elif any_reject:
        feedback_parts = [
            s["note"] for s in signals.values() if not s.get("accept", True)
        ]
        return BackstopResult(
            accepted=True,
            confidence="medium",
            signals=signals,
            feedback="; ".join(feedback_parts),
        )
    else:
        return BackstopResult(
            accepted=True, confidence="high", signals=signals, feedback=""
        )


# ---------------------------------------------------------------------------
# Helper functions (intensity / geometry signal extraction)
# ---------------------------------------------------------------------------


def _derive_bone_threshold(mask: np.ndarray, intensity: np.ndarray) -> float:
    """Otsu threshold on bone intensities."""
    tibia_intensities = intensity[mask > 0].astype(float)
    if tibia_intensities.size < 2:
        return 0.0
    return float(threshold_otsu(tibia_intensities))


def _compute_fill_ratios(
    mask: np.ndarray, intensity: np.ndarray, bone_threshold: float
) -> np.ndarray:
    """Per-slice bone fill ratio within the mask."""
    ratios = np.full(mask.shape[0], np.nan, dtype=float)
    for z in range(mask.shape[0]):
        total = int(mask[z].sum())
        if total == 0:
            continue
        bone_count = int(((intensity[z] > bone_threshold) & mask[z]).sum())
        ratios[z] = bone_count / total
    return ratios


def _intensity_gradient_signal(mask: np.ndarray, intensity: np.ndarray) -> int | None:
    """Find slice with largest negative intensity gradient spike."""
    mean_intensities = np.full(mask.shape[0], np.nan, dtype=float)
    for z in range(mask.shape[0]):
        vals = intensity[z][mask[z] > 0]
        if vals.size > 0:
            mean_intensities[z] = float(vals.mean())
    valid = ~np.isnan(mean_intensities)
    if valid.sum() < 5:
        return None
    grad = np.gradient(mean_intensities)
    grad[~valid] = 0
    # Find the largest negative spike after at least 5 stable slices
    min_idx = int(np.argmin(grad))
    if grad[min_idx] < -np.std(grad[valid]) * 2:
        return min_idx
    return None


def _texture_homogeneity_signal(
    mask: np.ndarray, intensity: np.ndarray
) -> int | None:
    """Find slice where intensity std within mask drops sustainedly."""
    stds = np.full(mask.shape[0], np.nan, dtype=float)
    for z in range(mask.shape[0]):
        vals = intensity[z][mask[z] > 0].astype(float)
        if vals.size > 5:
            stds[z] = float(vals.std())
    valid = ~np.isnan(stds)
    if valid.sum() < 5:
        return None
    # Find sustained drop (similar to fill ratio)
    median_std = float(np.nanmedian(stds))
    for z in range(mask.shape[0]):
        if not valid[z]:
            continue
        if stds[z] < median_std * 0.5:
            # Check if sustained
            below_count = 0
            for zz in range(z, min(z + 5, mask.shape[0])):
                if valid[zz] and stds[zz] < median_std * 0.5:
                    below_count += 1
            if below_count >= 3:
                return z
    return None


def _mask_area_gradient_signal(mask: np.ndarray) -> int | None:
    """Find slice with largest negative area gradient (geometry-only signal)."""
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float)
    if areas.max() == 0:
        return None
    grad = np.gradient(areas)
    # Find largest negative gradient after stable high region
    min_idx = int(np.argmin(grad))
    if grad[min_idx] < -areas.max() * 0.1:
        return min_idx
    return None
