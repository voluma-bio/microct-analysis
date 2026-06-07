"""Landmark placement and orientation correction stage driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu

from microct_analysis.domain.artifact_contracts import screenshot_path
from microct_analysis.processing.orientation import center_volume, pca_orient
from microct_analysis.processing.surface import (
    _condylar_si_limit,
    extract_surface_mesh,
    find_condylar_edge,
    find_notch_depth,
    find_saddle_point,
)
from microct_analysis.processing.types import LabelVolume

_AXIS_NAMES = {0: "superior-inferior", 1: "anterior-posterior", 2: "medial-lateral"}


def run_landmarks_orientation(
    segmentation_artifacts: dict[str, str],
    workflow_landmarks: list[dict[str, Any]],
    workflow_orientation: dict[str, Any],
    output_dir: str = "landmarks",
) -> dict[str, Any]:
    """Place workflow landmarks, compute orientation frame, and emit artifacts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = _load_label_volume(segmentation_artifacts.get("labels"))
    intensity = _load_intensity_volume(segmentation_artifacts, labels)
    assignments = _load_json(segmentation_artifacts.get("structure_assignments"))
    spacing = _spacing_from_artifacts(segmentation_artifacts, assignments)
    orientation_result = _orient_tibia(labels, intensity, assignments, workflow_landmarks, spacing, output_root)

    positions = {
        "landmarks": [
            _compute_landmark(
                definition,
                _labels_for_landmark(definition, labels, orientation_result),
                assignments,
                spacing,
                force_low_confidence=_force_low_tibial_confidence(definition, orientation_result),
                intensity=orientation_result.get("oriented_intensity"),
            )
            for definition in workflow_landmarks
        ],
        "coordinate_system": "volume_zyx",
        "spacing": list(spacing),
        "source_artifacts": dict(segmentation_artifacts),
        "orientation_applied": orientation_result["applied"],
    }

    orientation_frame = compute_orientation_frame(positions["landmarks"], workflow_orientation)
    _add_pca_orientation_provenance(orientation_frame, orientation_result, workflow_orientation)
    confidence, evidence = _landmark_confidence(positions["landmarks"], orientation_frame)
    _write_json(output_root / "positions.json", positions)
    _write_json(output_root / "orientation_frame.json", orientation_frame)
    _write_json(output_root / "transform_matrix.json", orientation_frame["pca_orientation"])
    _write_orientation_report(output_root / "orientation_report.md", orientation_frame, orientation_result)

    return {
        "stage": "landmarks",
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": _recommended_action(confidence),
        "artifacts": {
            "positions": str(output_root / "positions.json"),
            "orientation_frame": str(output_root / "orientation_frame.json"),
            "transform_matrix": str(output_root / "transform_matrix.json"),
            "oriented_labels": str(output_root / "oriented_labels.npy"),
            "orientation_report": str(output_root / "orientation_report.md"),
            "screenshots": [screenshot_path("landmarks", 1)],
        },
    }


def _orient_tibia(
    labels: np.ndarray | None,
    intensity: np.ndarray | None,
    assignments: dict[str, Any],
    workflow_landmarks: list[dict[str, Any]],
    spacing: tuple[float, float, float],
    output_root: Path,
) -> dict[str, Any]:
    tibial_requested = any(_is_tibial_landmark(definition) for definition in workflow_landmarks)
    tibia_label = _tibia_label(assignments, workflow_landmarks)
    identity = np.eye(3).round(8).tolist()
    fallback = {
        "applied": False,
        "confidence": "low" if tibial_requested else "high",
        "label": tibia_label,
        "rotation_matrix": identity,
        "translation": [0.0, 0.0, 0.0],
        "oriented_labels": labels,
        "oriented_intensity": intensity,
        "validation": "PCA orientation was not attempted.",
        "manual_instructions": _manual_orientation_instructions(),
    }
    if not tibial_requested:
        np.save(
            output_root / "oriented_labels.npy", labels if labels is not None else np.zeros((0, 0, 0), dtype=np.uint8)
        )
        return fallback | {"validation": "No tibial slice-boundary landmarks requested."}
    if labels is None or tibia_label is None:
        np.save(
            output_root / "oriented_labels.npy", labels if labels is not None else np.zeros((0, 0, 0), dtype=np.uint8)
        )
        return fallback | {"validation": "Missing label volume or tibia label assignment."}

    tibia_mask = labels == tibia_label
    source_intensity = intensity if intensity is not None else tibia_mask.astype(np.float32)
    try:
        _centered_mask, translation = center_volume(tibia_mask.astype(np.uint8), spacing)
        oriented_mask, oriented_intensity, rotation = pca_orient(tibia_mask.astype(np.uint8), source_intensity, spacing)
        validation = _validate_orientation(oriented_mask, rotation, labels.shape)
        if validation is not None:
            raise ValueError(validation)
        oriented_labels = np.array(labels, copy=True)
        oriented_labels[labels == tibia_label] = 0
        oriented_labels[oriented_mask != 0] = tibia_label
        np.save(output_root / "oriented_labels.npy", oriented_labels)
        return {
            "applied": True,
            "confidence": "high",
            "label": tibia_label,
            "rotation_matrix": np.asarray(rotation).round(8).tolist(),
            "translation": np.asarray(translation).round(8).tolist(),
            "oriented_labels": oriented_labels,
            "oriented_intensity": oriented_intensity,
            "validation": "PCA orientation passed shape, foreground, and rotation-matrix checks.",
            "manual_instructions": None,
        }
    except Exception as exc:
        np.save(output_root / "oriented_labels.npy", labels)
        return fallback | {"validation": f"PCA orientation failed: {exc}"}


def _validate_orientation(oriented_mask: np.ndarray, rotation: np.ndarray, shape: tuple[int, ...]) -> str | None:
    if oriented_mask.shape != shape:
        return "oriented tibia mask changed shape"
    if np.count_nonzero(oriented_mask) < 3:
        return "oriented tibia mask has fewer than three foreground voxels"
    rotation_array = np.asarray(rotation, dtype=float)
    if rotation_array.shape != (3, 3) or not np.all(np.isfinite(rotation_array)):
        return "rotation matrix is not finite 3x3"
    if not np.allclose(rotation_array @ rotation_array.T, np.eye(3), atol=1e-5):
        return "rotation matrix is not orthonormal"
    determinant = float(np.linalg.det(rotation_array))
    if not np.isclose(abs(determinant), 1.0, atol=1e-5):
        return "rotation matrix determinant is implausible"
    return None


def _labels_for_landmark(
    definition: dict[str, Any], labels: np.ndarray | None, orientation_result: dict[str, Any]
) -> np.ndarray | None:
    if _is_tibial_landmark(definition) and orientation_result.get("applied"):
        return orientation_result.get("oriented_labels")
    return labels


def _force_low_tibial_confidence(definition: dict[str, Any], orientation_result: dict[str, Any]) -> bool:
    return _is_tibial_landmark(definition) and not bool(orientation_result.get("applied"))


def _is_tibial_landmark(definition: dict[str, Any]) -> bool:
    structure = str(
        definition.get("structure") or definition.get("target_structure") or definition.get("bone") or ""
    ).lower()
    return str(definition.get("domain", "")) == "tibial_2d_slice" or "tibia" in structure


def _tibia_label(assignments: dict[str, Any], workflow_landmarks: list[dict[str, Any]]) -> int | None:
    for name in ("tibia", "Tibia"):
        value = _label_for_structure(name, assignments, {})
        if value is not None:
            return value
    for definition in workflow_landmarks:
        if _is_tibial_landmark(definition):
            return _label_for_structure("", assignments, definition)
    return None


def _add_pca_orientation_provenance(
    orientation_frame: dict[str, Any], orientation_result: dict[str, Any], workflow_orientation: dict[str, Any]
) -> None:
    rotation = orientation_result["rotation_matrix"]
    translation = orientation_result["translation"]
    axes = {
        "superior_inferior": rotation[0],
        "medial_lateral": rotation[1],
        "anterior_posterior": rotation[2],
    }
    orientation_frame["pca_orientation"] = {
        "type": "pca-centered-rigid-orientation",
        "applied": orientation_result["applied"],
        "confidence": orientation_result["confidence"],
        "tibia_label": orientation_result["label"],
        "translation": translation,
        "rotation_matrix": rotation,
        "preserve_voxel_size": True,
        "label_interpolation_order": 0,
        "intensity_interpolation_order": 1,
        "validation": orientation_result["validation"],
        "explanation": explain_axis_changes(axes, str(workflow_orientation.get("target_plane", "frontal"))),
    }
    if orientation_result.get("manual_instructions"):
        orientation_frame["pca_orientation"]["manual_instructions"] = orientation_result["manual_instructions"]
    orientation_frame["rotation_matrix"] = rotation
    orientation_frame["translation"] = translation
    orientation_frame["orientation_confidence"] = orientation_result["confidence"]
    orientation_frame["explanation"] = orientation_frame["pca_orientation"]["explanation"]


def _manual_orientation_instructions() -> str:
    return (
        "PCA orientation could not be validated. Open the tibia label/intensity volume in a PyVista "
        "interactive session, center the tibia on its bounding-box midpoint, rotate until the frontal "
        "plane matches the SOP xz view, resample with voxel size preserved, then rerun landmark placement."
    )


def _write_orientation_report(
    path: Path, orientation_frame: dict[str, Any], orientation_result: dict[str, Any]
) -> None:
    pca = orientation_frame["pca_orientation"]
    lines = [
        "# Landmark Orientation Report",
        "",
        f"Confidence: {pca['confidence']}",
        f"Applied: {pca['applied']}",
        f"Validation: {pca['validation']}",
        "",
        f"Explanation: {pca['explanation']}",
        "",
        f"Translation: {pca['translation']}",
        f"Rotation matrix: {pca['rotation_matrix']}",
    ]
    if orientation_result.get("manual_instructions"):
        lines.extend(["", "## User-assisted fallback", "", orientation_result["manual_instructions"]])
    path.write_text("\n".join(lines) + "\n")


def compute_orientation_frame(landmarks: list[dict[str, Any]], workflow_orientation: dict[str, Any]) -> dict[str, Any]:
    """Compute target orientation vectors and plain-language axis-change explanation."""

    by_id = {item["id"]: np.asarray(item["physical"], dtype=float) for item in landmarks}
    axes: dict[str, list[float]] = {}
    for axis_name, spec in (workflow_orientation.get("axes") or {}).items():
        vector = _axis_vector(spec, by_id)
        axes[axis_name] = _unit(vector).round(8).tolist()

    if not axes:
        axes = {
            "superior_inferior": [1.0, 0.0, 0.0],
            "anterior_posterior": [0.0, 1.0, 0.0],
            "medial_lateral": [0.0, 0.0, 1.0],
        }

    target_plane = str(workflow_orientation.get("target_plane", "frontal"))
    transform = {
        "type": "landmark-derived-rigid-orientation",
        "target_plane": target_plane,
        "axes": axes,
        "translation": _translation(workflow_orientation, by_id),
        "rotation_matrix": _rotation_matrix(axes),
        "explanation": explain_axis_changes(axes, target_plane),
        "source_landmarks": sorted(by_id),
    }
    return transform


def explain_axis_changes(axes: dict[str, list[float]], target_plane: str) -> str:
    """Explain orientation correction in operator-facing language."""

    pieces = [f"Aligned the specimen to the workflow {target_plane} plane."]
    for axis_name, vector in axes.items():
        dominant = int(np.argmax(np.abs(np.asarray(vector, dtype=float))))
        direction = "positive" if vector[dominant] >= 0 else "negative"
        readable = axis_name.replace("_", "-")
        pieces.append(f"{readable} now follows the {direction} {_AXIS_NAMES[dominant]} volume direction.")
    return " ".join(pieces)


def _compute_landmark(
    definition: dict[str, Any],
    labels: np.ndarray | None,
    assignments: dict[str, Any],
    spacing: tuple[float, float, float],
    force_low_confidence: bool = False,
    intensity: np.ndarray | None = None,
) -> dict[str, Any]:
    landmark_id = str(definition.get("id") or definition.get("name"))
    structure = str(definition.get("structure") or definition.get("target_structure") or definition.get("bone") or "")
    method = str(definition.get("geometric_method") or definition.get("method", "centroid"))
    domain = str(definition.get("domain", ""))
    label_value = _label_for_structure(structure, assignments, definition)
    voxel, confidence, note = _position_from_definition(definition, labels, label_value, spacing, method, intensity)
    if force_low_confidence:
        confidence = "low"
        note = f"PCA orientation unavailable; user-assisted PyVista orientation required. {note}"
    physical = tuple(float(voxel[index]) * spacing[index] for index in range(3))
    return {
        "id": landmark_id,
        "structure": structure,
        "domain": domain or None,
        "method": method,
        "label": label_value,
        "voxel": [float(value) for value in voxel],
        "physical": [float(value) for value in physical],
        "confidence": confidence,
        "requires_user_confirmation": confidence == "low",
        "evidence": note,
    }


def _position_from_definition(
    definition: dict[str, Any],
    labels: np.ndarray | None,
    label_value: int | None,
    spacing: tuple[float, float, float],
    method: str,
    intensity: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], str, str]:
    if "voxel" in definition:
        return _triple(definition["voxel"]), "high", "landmark provided explicit voxel coordinates"
    if "physical" in definition:
        physical = _triple(definition["physical"])
        return (
            tuple(physical[index] / spacing[index] for index in range(3)),
            "high",
            "landmark provided explicit physical coordinates",
        )
    if labels is None or label_value is None:
        return (0.0, 0.0, 0.0), "medium", "missing label volume or structure label; used origin fallback"

    domain = str(definition.get("domain", ""))
    mask = labels == label_value
    if domain == "femoral_3d_surface":
        return _femoral_surface_position(definition, mask, spacing, method)
    if domain == "tibial_2d_slice":
        return _tibial_slice_position(definition, mask, spacing, method, intensity)

    return _centroid_fallback(mask, method, "legacy label-statistic landmark")


def _femoral_surface_position(
    definition: dict[str, Any], mask: np.ndarray, spacing: tuple[float, float, float], method: str
) -> tuple[tuple[float, float, float], str, str]:
    try:
        vertices, _faces = extract_surface_mesh(mask, spacing)
        label_volume = LabelVolume(data=mask.astype(np.uint8), spacing=spacing, label_map={"foreground": 1})
        if len(vertices) < 4:
            raise ValueError("surface mesh has too few vertices")
        landmark_id = str(definition.get("id") or definition.get("name"))
        params = definition.get("geometric_params") if isinstance(definition.get("geometric_params"), dict) else {}
        if method == "saddle_point" or landmark_id == "intercondylar_groove_midpoint":
            point, _scores, _vertices = find_saddle_point(vertices, surface_region=str(params.get("surface_region", "anterior_distal")))
        elif method == "notch_depth_maximum" or landmark_id == "intercondylar_notch":
            point, _scores, _vertices = find_notch_depth(vertices, surface_region=str(params.get("surface_region", "posterior_intercondylar")))
        elif method == "surface_extreme" or landmark_id in {"lateral_condylar_edge", "medial_condylar_edge"}:
            direction = str(params.get("direction") or ("lateral" if "lateral" in landmark_id else "medial"))
            point = find_condylar_edge(
                vertices, direction, include_osteophytes=bool(params.get("include_osteophytes", True))
            )
        else:
            raise ValueError(f"unsupported femoral surface landmark method: {method}")
        confidence = "high"
        note = f"surface feature detected from {len(vertices)} mesh vertices"
        if landmark_id == "intercondylar_notch":
            posterior = vertices[vertices[:, 1] > np.median(vertices[:, 1])]
            si_range = np.ptp(vertices[:, 0])
            si_min = np.min(vertices[:, 0])
            relative_position = (point[0] - si_min) / si_range if si_range > 0 else 0.5
            if point[0] > _condylar_si_limit(posterior) or relative_position > 0.95:
                confidence = "low"
                note += "; notch position is implausibly proximal (shaft region)"
        voxel = tuple(float(point[index]) / label_volume.spacing[index] for index in range(3))
        return voxel, confidence, note
    except Exception as exc:
        voxel, _confidence, _note = _centroid_fallback(mask, method, f"surface detection failed: {exc}")
        return voxel, "low", f"surface detection failed; centroid/extrema fallback used: {exc}"


def _tibial_slice_position(
    definition: dict[str, Any],
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    method: str,
    intensity: np.ndarray | None = None,
) -> tuple[tuple[float, float, float], str, str]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0), "low", "tibia label has no foreground voxels"
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    landmark_id = str(definition.get("id") or definition.get("name"))
    params = definition.get("geometric_params") if isinstance(definition.get("geometric_params"), dict) else {}
    articular = _articular_slice(counts, float(params.get("area_threshold_pct", 20)))
    growth, growth_source = _growth_plate_slice_result(mask, counts, articular, params, intensity)
    measurement_slice = _measurement_slice(mask, articular, growth)
    if landmark_id == "articular_surface_proximal" or params.get("surface") == "articular":
        z_index = articular
        return _slice_center(mask, z_index), "high", f"articular boundary at slice {z_index}"
    if landmark_id == "growth_plate_proximal" or params.get("detection") == "bone_fill_ratio_drop":
        confidence = "high" if growth is not None else "low"
        z_index = growth if growth is not None else int(coords[:, 0].max())
        validation = "sustained-drop validated" if growth_source == "intensity" else "label-area fallback"
        return (
            _slice_center(mask, z_index),
            confidence,
            f"growth plate boundary at slice {z_index}; {validation}",
        )
    if method == "slice_bone_extent" or landmark_id in {"medial_tibial_condyle_edge", "lateral_tibial_condyle_edge"}:
        direction = str(params.get("direction") or ("lateral" if "lateral" in landmark_id else "medial"))
        return (
            _slice_edge(mask, measurement_slice, direction),
            "high",
            f"{direction} tibial edge on measurement slice {measurement_slice}",
        )
    return _centroid_fallback(mask, method, "unsupported tibial slice landmark fallback")


def _centroid_fallback(mask: np.ndarray, method: str, note: str) -> tuple[tuple[float, float, float], str, str]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0), "low", f"{note}; empty structure mask"
    if method in {"superior", "proximal"}:
        return _triple(coords[np.argmin(coords[:, 0])]), "high", note
    if method in {"inferior", "distal", "growth_plate"}:
        return _triple(coords[np.argmax(coords[:, 0])]), "high", note
    return _triple(coords.mean(axis=0)), "high", note


def _articular_slice(counts: np.ndarray, threshold_pct: float) -> int:
    nonzero = np.flatnonzero(counts)
    if nonzero.size == 0:
        return 0
    threshold = max(float(counts.max()) * (threshold_pct / 100.0), 1.0)
    candidates = np.flatnonzero(counts >= threshold)
    return int(candidates[0]) if candidates.size else int(nonzero[0])


def _growth_plate_slice(
    mask: np.ndarray,
    counts: np.ndarray,
    articular: int,
    params: dict[str, Any],
    intensity: np.ndarray | None = None,
) -> int | None:
    result, _source = _growth_plate_slice_result(mask, counts, articular, params, intensity)
    return result


def _growth_plate_slice_result(
    mask: np.ndarray,
    counts: np.ndarray,
    articular: int,
    params: dict[str, Any],
    intensity: np.ndarray | None = None,
) -> tuple[int | None, str | None]:
    if str(params.get("detection", "")) == "bone_fill_ratio_drop" and intensity is not None:
        result = _growth_plate_slice_intensity(mask, intensity, articular, params)
        if result is not None:
            return result, "intensity"

    min_consecutive = int(params.get("min_consecutive_above", 5))
    # Label-only fallback: first pronounced area drop after a stable tibial plateau.
    max_count = float(counts.max())
    if max_count == 0:
        return None, None
    above_seen = 0
    drop_threshold = max_count * 0.5
    candidates: list[int] = []
    for z_index in range(articular, len(counts)):
        if counts[z_index] >= drop_threshold:
            above_seen += 1
            continue
        if above_seen >= min_consecutive and counts[z_index] > 0:
            candidates.append(z_index)
    return (min(candidates), "label_area") if candidates else (None, None)


def _growth_plate_slice_intensity(
    mask: np.ndarray,
    intensity: np.ndarray,
    articular: int,
    params: dict[str, Any],
) -> int | None:
    bone_threshold = _derive_bone_threshold(mask, intensity)
    fill_threshold = float(params.get("fill_ratio_threshold_pct", 50)) / 100.0
    min_consecutive = int(params.get("min_consecutive_above", 5))
    min_sustained = int(params.get("min_sustained_below", 3))

    ratios = np.full(mask.shape[0], np.nan, dtype=float)
    for z_index in range(articular, mask.shape[0]):
        slice_mask = mask[z_index]
        total_label = int(slice_mask.sum())
        if total_label == 0:
            continue

        bone_count = int(((intensity[z_index] > bone_threshold) & slice_mask).sum())
        ratios[z_index] = bone_count / total_label

    return _growth_plate_slice_from_ratios(ratios, articular, fill_threshold, min_consecutive, min_sustained)


def _growth_plate_slice_from_ratios(
    ratios: np.ndarray,
    articular: int,
    fill_threshold: float,
    min_consecutive: int,
    min_sustained: int,
) -> int | None:
    above_seen = 0
    for z_index in range(articular, len(ratios)):
        if np.isnan(ratios[z_index]):
            above_seen = 0
            continue
        if ratios[z_index] >= fill_threshold:
            above_seen += 1
        else:
            if above_seen >= min_consecutive and _is_sustained_drop(ratios, z_index, fill_threshold, min_sustained):
                return z_index
            above_seen = 0
    return None


def _is_sustained_drop(ratios: np.ndarray, start: int, threshold: float, n_required: int) -> bool:
    below_count = 0
    for z_index in range(start, len(ratios)):
        if np.isnan(ratios[z_index]):
            continue
        if ratios[z_index] < threshold:
            below_count += 1
            if below_count >= n_required:
                return True
        else:
            return False
    return below_count > 0


def _derive_bone_threshold(mask: np.ndarray, intensity: np.ndarray) -> float:
    tibia_intensities = intensity[mask > 0].astype(float)
    if tibia_intensities.size < 2:
        return 0.0
    return float(threshold_otsu(tibia_intensities))


def _measurement_slice(mask: np.ndarray, articular: int, growth: int | None) -> int:
    stop = growth if growth is not None else mask.shape[0] - 1
    start, stop = sorted((max(0, articular), min(mask.shape[0] - 1, stop)))
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    region = np.arange(start, stop + 1)
    max_area = int(areas[region].max()) if region.size else 0
    tied = region[areas[region] >= max_area - 2]
    return int(tied[-1]) if tied.size else start


def _slice_center(mask: np.ndarray, z_index: int) -> tuple[float, float, float]:
    coords = np.argwhere(mask[z_index])
    if coords.size == 0:
        all_coords = np.argwhere(mask)
        yx = all_coords[:, 1:].mean(axis=0)
    else:
        yx = coords.mean(axis=0)
    return (float(z_index), float(yx[0]), float(yx[1]))


def _slice_edge(mask: np.ndarray, z_index: int, direction: str) -> tuple[float, float, float]:
    coords = np.argwhere(mask[z_index])
    if coords.size == 0:
        return _slice_center(mask, z_index)
    edge = coords[np.argmax(coords[:, 1])] if direction == "lateral" else coords[np.argmin(coords[:, 1])]
    return (float(z_index), float(edge[0]), float(edge[1]))


def _axis_vector(spec: Any, landmarks: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(spec, dict) and "from" in spec and "to" in spec:
        return landmarks[str(spec["to"])] - landmarks[str(spec["from"])]
    if isinstance(spec, dict) and "vector" in spec:
        return np.asarray(spec["vector"], dtype=float)
    if isinstance(spec, (list, tuple)):
        return np.asarray(spec, dtype=float)
    raise ValueError(f"unsupported orientation axis spec: {spec!r}")


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("orientation axis vector has zero length")
    return vector / norm


def _rotation_matrix(axes: dict[str, list[float]]) -> list[list[float]]:
    rows = list(axes.values())[:3]
    identity = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    while len(rows) < 3:
        rows.append(identity[len(rows)])
    return rows


def _translation(workflow_orientation: dict[str, Any], landmarks: dict[str, np.ndarray]) -> list[float]:
    origin_id = workflow_orientation.get("origin_landmark")
    if origin_id and str(origin_id) in landmarks:
        return (-landmarks[str(origin_id)]).round(8).tolist()
    return [0.0, 0.0, 0.0]


def _landmark_confidence(landmarks: list[dict[str, Any]], orientation_frame: dict[str, Any]) -> tuple[str, str]:
    if orientation_frame.get("orientation_confidence") == "low":
        return "low", "PCA orientation unavailable; tibial landmarks require user-assisted PyVista orientation."
    tibial_issue = _tibial_interval_issue(landmarks)
    if tibial_issue is not None:
        return "medium", tibial_issue
    weak = [item["id"] for item in landmarks if item.get("confidence") != "high"]
    if weak:
        return "medium", f"Computed landmarks, but {', '.join(weak)} used fallback coordinates."
    return (
        "high",
        "Landmarks placed from workflow definitions and orientation transform recorded. "
        + orientation_frame["explanation"],
    )


def _tibial_interval_issue(landmarks: list[dict[str, Any]]) -> str | None:
    articular = next((item for item in landmarks if "articular" in str(item.get("id", ""))), None)
    growth = next((item for item in landmarks if "growth_plate" in str(item.get("id", ""))), None)
    if articular is None or growth is None:
        return None

    growth_z = float(growth.get("voxel", [0.0])[0])
    articular_z = float(articular.get("voxel", [0.0])[0])
    iioc_slices = growth_z - articular_z
    if iioc_slices <= 0 or iioc_slices > 100:
        growth["confidence"] = "medium"
        growth["evidence"] = (
            f"Growth plate at slice {growth_z:g} is implausible (IIOC = {iioc_slices:g} slices); "
            "review landmark detection."
        )
        return growth["evidence"]
    return None


def _recommended_action(confidence: str) -> str:
    return {"high": "proceed", "medium": "flag", "low": "pause"}[confidence]


def _load_intensity_volume(segmentation_artifacts: dict[str, str], labels: np.ndarray | None) -> np.ndarray | None:
    for key in ("intensity", "intensity_volume", "filtered", "volume"):
        loaded = _load_label_volume(segmentation_artifacts.get(key))
        if loaded is not None:
            return loaded
    return None


def _load_label_volume(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    label_path = Path(path)
    if not label_path.exists():
        return None
    if label_path.suffix == ".npy":
        return np.load(label_path)
    if label_path.suffix == ".npz":
        data = np.load(label_path)
        return data[data.files[0]]
    if label_path.suffix == ".json":
        return np.asarray(json.loads(label_path.read_text()))
    if label_path.name.endswith((".nii", ".nii.gz")):
        return np.asarray(nib.load(str(label_path)).get_fdata(), dtype=np.uint16)
    return None


def _load_json(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def _spacing_from_artifacts(
    segmentation_artifacts: dict[str, str], assignments: dict[str, Any]
) -> tuple[float, float, float]:
    raw = segmentation_artifacts.get("spacing") or assignments.get("spacing") or (1.0, 1.0, 1.0)
    return _triple(raw)


def _label_for_structure(structure: str, assignments: dict[str, Any], definition: dict[str, Any]) -> int | None:
    if "label" in definition:
        return int(definition["label"])
    mapping = assignments.get("assignments", assignments)
    lookup = structure or str(definition.get("bone") or "")
    value = mapping.get(lookup) if isinstance(mapping, dict) else None
    return int(value) if isinstance(value, int | float | str) and str(value).lstrip("-").isdigit() else None


def _triple(raw: Any) -> tuple[float, float, float]:
    values = list(raw)
    if len(values) != 3:
        raise ValueError("expected three coordinate values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
