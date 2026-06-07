from __future__ import annotations

import math
from pathlib import Path

from microct_analysis.measurements.geometry import (
    compute_boundary_slice_count,
    compute_frontal_projected_width,
    compute_ratio,
    compute_slice_distance,
    compute_surface_distance,
)
from microct_analysis.measurements.models import MeasurementResult, MeasurementSpec
from microct_analysis.measurements.reporting import build_qc_payload
from microct_analysis.measurements.workflow_binding import compile_measurement_specs
from microct_analysis.workflows.loading import load_workflow


WORKFLOW_PATH = Path("tests/fixtures/workflows/mouse-knee-oa-geometric-indices/workflow.md")


def _within_pct(value: float, expected: float, pct: float = 0.10) -> bool:
    return expected * (1 - pct) <= value <= expected * (1 + pct)


def test_oa6_femoral_surface_measurements_and_ratio_match_published_acceptance() -> None:
    landmarks = {
        "intercondylar_groove_midpoint": [0.0, 0.0, 0.0],
        "intercondylar_notch": [2.29, 0.0, 0.0],
        "lateral_condylar_edge": [0.0, 2.0, 3.48],
        "medial_condylar_edge": [0.0, -4.0, 0.0],
        "_derived_frame": {"ml_vector": [0.0, 0.0, 1.0]},
    }
    results = _run_specs(
        _oa6_specs({"distal_femoral_length", "distal_femoral_width", "distal_femoral_ratio"}),
        landmarks,
        (1.0, 1.0, 1.0),
    )
    length = results["distal_femoral_length"]
    width = results["distal_femoral_width"]
    ratio = results["distal_femoral_ratio"]

    assert _within_pct(length.value, 2.29)
    assert _within_pct(width.value, 3.48)
    assert _within_pct(ratio.value, 1.520)
    assert width.inputs["method"] == "frontal_projected_width"
    assert width.inputs["projection_method"] == "landmark_derived_ml_vector"


def test_oa6_tibial_slice_measurements_are_domain_routed_and_exact_slice_multiple() -> None:
    landmarks = {
        "articular_surface_proximal": {"slice_index": 661},
        "growth_plate_proximal": {"slice_index": 732},
        "medial_tibial_condyle_edge": [700.0, 0.0, 100.0],
        "lateral_tibial_condyle_edge": [700.0, 0.0, 380.9523809524],
    }
    spacing = (0.0105, 0.0105, 0.0105)
    results = _run_specs(
        _oa6_specs({"tibial_iioc_height", "tibial_width", "tibial_iioc_ratio"}),
        landmarks,
        spacing,
    )
    height = results["tibial_iioc_height"]
    width = results["tibial_width"]
    ratio = results["tibial_iioc_ratio"]
    qc = build_qc_payload([height, width, ratio])["qc_overlays"]

    assert height.inputs["slice_count"] == 71
    assert height.value == 71 * 0.0105
    assert math.isclose(height.value / 0.0105, round(height.value / 0.0105))
    assert height.inputs["exact_multiple"] is True
    assert _within_pct(width.value, 2.95)
    assert _within_pct(ratio.value, 0.253)
    assert qc[0]["domain"] == "tibial_2d_slice"
    assert qc[0]["method"] == "boundary_slice_count"
    assert qc[0]["slice_count"] == 71
    assert qc[0]["exact_multiple"] is True


def _oa6_specs(wanted: set[str]) -> list[MeasurementSpec]:
    workflow = load_workflow(WORKFLOW_PATH)
    specs = compile_measurement_specs(workflow)
    return [spec for spec in specs if spec.name in wanted]


def _run_specs(
    specs: list[MeasurementSpec], landmarks: dict[str, object], spacing: tuple[float, ...]
) -> dict[str, MeasurementResult]:
    results: dict[str, MeasurementResult] = {}
    for spec in specs:
        if spec.kind == "surface_distance":
            result = compute_surface_distance(spec, landmarks, spacing)
        elif spec.kind == "frontal_projected_width":
            result = compute_frontal_projected_width(spec, landmarks, spacing)
        elif spec.kind == "boundary_slice_count":
            result = compute_boundary_slice_count(spec, landmarks, spacing)
        elif spec.kind == "slice_distance":
            result = compute_slice_distance(spec, landmarks, spacing)
        elif spec.kind == "ratio":
            result = compute_ratio(spec, results)
        else:
            continue
        results[result.name] = result
    return results
