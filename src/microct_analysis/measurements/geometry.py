"""Geometric distance, ratio, and slice-count measurement primitives."""

from __future__ import annotations

import math
import numpy as np
from typing import Any

from .models import MeasurementResult, MeasurementSpec


def compute_distance(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Compute Euclidean distance between two landmark points."""

    if not spec.points or len(spec.points) != 2:
        raise ValueError(f"distance measurement {spec.name} requires exactly two points")
    first_name, second_name = spec.points
    first = _point(landmarks, first_name)
    second = _point(landmarks, second_name)
    value = math.sqrt(sum(((second[i] - first[i]) * _axis_spacing(spacing, i)) ** 2 for i in range(len(first))))
    return MeasurementResult(spec.name, value, spec.unit, spec, {"points": {first_name: first, second_name: second}, "spacing": list(spacing)}, f"measurements/qc/{spec.name}.json")



def compute_surface_distance(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Compute 3D Euclidean distance between two surface landmark points."""

    result = compute_distance(spec, landmarks, spacing)
    return _with_method(result, "surface_distance", "euclidean_3d")


def compute_slice_distance(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Compute 2D distance on an oriented slice between two boundary points."""

    if not spec.points or len(spec.points) != 2:
        raise ValueError(f"slice-distance measurement {spec.name} requires exactly two points")
    first_name, second_name = spec.points
    first = _point(landmarks, first_name)
    second = _point(landmarks, second_name)
    axes = _slice_axes(spec, len(first))
    value = math.sqrt(sum(((second[i] - first[i]) * _axis_spacing(spacing, i)) ** 2 for i in axes))
    return MeasurementResult(
        spec.name,
        value,
        spec.unit,
        spec,
        {
            "points": {first_name: first, second_name: second},
            "spacing": list(spacing),
            "domain": spec.domain,
            "method": "bone_extent_on_slice",
            "slice_axes": list(axes),
        },
        f"measurements/qc/{spec.name}.json",
    )


def compute_boundary_slice_count(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Count Z slices between two boundaries and multiply by slice thickness."""

    if not spec.boundaries or len(spec.boundaries) != 2:
        raise ValueError(f"boundary-slice-count measurement {spec.name} requires exactly two boundaries")
    start_name, end_name = spec.boundaries
    start_index = _slice_index(landmarks, start_name)
    end_index = _slice_index(landmarks, end_name)
    slice_count = abs(end_index - start_index)
    slice_thickness = _slice_thickness(spec, spacing)
    value = slice_count * slice_thickness
    return MeasurementResult(
        spec.name,
        value,
        spec.unit,
        spec,
        {
            "boundaries": {start_name: start_index, end_name: end_index},
            "slice_count": slice_count,
            "slice_thickness": slice_thickness,
            "exact_multiple": True,
            "domain": spec.domain,
            "method": "boundary_slice_count",
        },
        f"measurements/qc/{spec.name}.json",
    )


def compute_frontal_projected_width(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Compute medial-lateral width on the frontal projection using landmark-derived ML vector."""

    if not spec.points or len(spec.points) != 2:
        raise ValueError(f"frontal-projected-width measurement {spec.name} requires exactly two points")
    first_name, second_name = spec.points
    first = _point(landmarks, first_name)
    second = _point(landmarks, second_name)

    # Use landmark-derived ML vector — no silent fallback
    derived_frame = None
    if isinstance(landmarks.get("_derived_frame"), dict):
        derived_frame = landmarks["_derived_frame"]

    if derived_frame is not None and "ml_vector" in derived_frame:
        ml_vector = np.asarray(derived_frame["ml_vector"], dtype=float)
        ml_norm = float(np.linalg.norm(ml_vector))
        if ml_norm == 0:
            raise ValueError("landmark-derived ML vector has zero length")
        ml_unit = ml_vector / ml_norm
        delta_physical = np.array([
            (second[i] - first[i]) * _axis_spacing(spacing, i) for i in range(len(first))
        ])
        value = abs(float(np.dot(delta_physical, ml_unit)))
    else:
        raise ValueError("landmark-derived ML vector required for frontal projected width")

    return MeasurementResult(
        spec.name,
        value,
        spec.unit,
        spec,
        {
            "points": {first_name: first, second_name: second},
            "spacing": list(spacing),
            "domain": spec.domain,
            "method": "frontal_projected_width",
            "projection_method": "landmark_derived_ml_vector",
        },
        f"measurements/qc/{spec.name}.json",
    )

def compute_ratio(spec: MeasurementSpec, component_results: dict[str, MeasurementResult]) -> MeasurementResult:
    """Compute ratio from two component measurements."""

    if not spec.numerator or not spec.denominator:
        raise ValueError(f"ratio measurement {spec.name} requires numerator and denominator")
    numerator = component_results[spec.numerator]
    denominator = component_results[spec.denominator]
    if denominator.value == 0:
        raise ValueError(f"ratio measurement {spec.name} denominator is zero")
    return MeasurementResult(spec.name, numerator.value / denominator.value, spec.unit, spec, {"numerator": spec.numerator, "denominator": spec.denominator}, f"measurements/qc/{spec.name}.json")


def compute_slice_count(spec: MeasurementSpec, landmarks: dict[str, Any], spacing: tuple[float, ...]) -> MeasurementResult:
    """Compute distance from slice count x slice thickness."""

    if not spec.boundaries or len(spec.boundaries) != 2:
        raise ValueError(f"slice-count measurement {spec.name} requires exactly two boundaries")
    start_name, end_name = spec.boundaries
    start = _point(landmarks, start_name)
    end = _point(landmarks, end_name)
    axis = max(range(len(start)), key=lambda index: abs(end[index] - start[index]))
    value = abs(end[axis] - start[axis]) * _axis_spacing(spacing, axis)
    return MeasurementResult(spec.name, value, spec.unit, spec, {"boundaries": {start_name: start, end_name: end}, "axis": axis, "spacing": list(spacing)}, f"measurements/qc/{spec.name}.json")


def _point(landmarks: dict[str, Any], name: str) -> tuple[float, ...]:
    raw = landmarks.get(name)
    if raw is None and "landmarks" in landmarks:
        for item in landmarks["landmarks"]:
            if isinstance(item, dict) and item.get("id") == name:
                raw = item.get("physical") or item.get("voxel")
                break
    if raw is None:
        raise KeyError(f"missing landmark {name!r}")
    values = list(raw)
    if not values:
        raise ValueError(f"landmark {name!r} is empty")
    return tuple(float(value) for value in values)


def _axis_spacing(spacing: tuple[float, ...], axis: int) -> float:
    return float(spacing[axis]) if axis < len(spacing) else 1.0


def _with_method(result: MeasurementResult, kind: str, method: str) -> MeasurementResult:
    inputs = dict(result.inputs)
    inputs.update({"domain": result.spec.domain, "method": method, "measurement_kind": kind})
    return MeasurementResult(result.name, result.value, result.unit, result.spec, inputs, result.qc_evidence)


def _slice_axes(spec: MeasurementSpec, ndim: int) -> tuple[int, ...]:
    if spec.projection in {"frontal", "coronal"}:
        return tuple(axis for axis in (0, 2) if axis < ndim) or (0,)
    return tuple(range(1, ndim)) if ndim > 1 else (0,)


def _projection_axis(spec: MeasurementSpec) -> int:
    return 2 if spec.projection in {"frontal", "coronal"} else 0


def _slice_index(landmarks: dict[str, Any], name: str) -> int:
    raw = landmarks.get(name)
    if raw is None and "landmarks" in landmarks:
        for item in landmarks["landmarks"]:
            if isinstance(item, dict) and item.get("id") == name:
                raw = item
                break
    if isinstance(raw, dict):
        for field in ("slice_index", "z_index", "index"):
            if field in raw:
                return int(raw[field])
        raw = raw.get("voxel") or raw.get("physical")
    if raw is None:
        raise KeyError(f"missing boundary landmark {name!r}")
    values = list(raw) if not isinstance(raw, (str, bytes)) else [raw]
    if len(values) == 1:
        return int(values[0])
    return int(values[0])


def _slice_thickness(spec: MeasurementSpec, spacing: tuple[float, ...]) -> float:
    if spec.slice_thickness_mm is not None:
        return float(spec.slice_thickness_mm)
    if spec.acceptance and "slice_thickness_mm" in spec.acceptance:
        return float(spec.acceptance["slice_thickness_mm"])
    return float(spacing[0]) if spacing else 0.0105
