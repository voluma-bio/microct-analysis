"""Measurement stage driver — executed via jupyter-workbench exec --file."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from microct_analysis.domain.artifact_contracts import screenshot_path
from microct_analysis.measurements.geometry import (
    compute_boundary_slice_count,
    compute_distance,
    compute_frontal_projected_width,
    compute_ratio,
    compute_slice_count,
    compute_slice_distance,
    compute_surface_distance,
)
from microct_analysis.measurements.models import MeasurementResult, OverrideRecord
from microct_analysis.measurements.reporting import build_qc_payload, results_to_json, results_to_markdown
from microct_analysis.measurements.trabecular import compute_trabecular_metrics
from microct_analysis.measurements.volumes import compute_labeled_volume
from microct_analysis.measurements.workflow_binding import compile_measurement_specs


def run_measurement(
    landmark_artifacts: dict[str, str],
    roi_artifacts: dict[str, str],
    segmentation_artifacts: dict[str, str],
    workflow_measurements: list[dict[str, Any]],
    workflow_roi_defs: list[dict[str, Any]],
    spacing: tuple[float, ...],
    output_dir: str = "measurements",
) -> dict[str, Any]:
    """Execute all workflow-defined measurements.

    Returns stage report with results.json, qc_overlays.json, overrides.json.
    Records per-run overrides without mutating canonical workflow.
    """

    output_root = Path(output_dir)
    (output_root / "qc").mkdir(parents=True, exist_ok=True)
    landmarks = _load_landmarks(landmark_artifacts)
    labels = _load_array(segmentation_artifacts.get("labels"))
    roi_masks = _load_roi_masks(roi_artifacts)
    specs = compile_measurement_specs({"measurements": workflow_measurements})
    raw_by_name = {str(item.get("name")): item for item in workflow_measurements}

    results: list[MeasurementResult] = []
    by_name: dict[str, MeasurementResult] = {}
    for spec in specs:
        raw = raw_by_name.get(spec.name, {})
        if spec.kind == "distance":
            result = compute_distance(spec, landmarks, spacing)
        elif spec.kind == "surface_distance":
            result = compute_surface_distance(spec, landmarks, spacing)
        elif spec.kind == "slice_distance":
            result = compute_slice_distance(spec, landmarks, spacing)
        elif spec.kind == "slice_count":
            result = compute_slice_count(spec, landmarks, spacing)
        elif spec.kind == "boundary_slice_count":
            result = compute_boundary_slice_count(spec, landmarks, spacing)
        elif spec.kind == "frontal_projected_width":
            result = compute_frontal_projected_width(spec, landmarks, spacing)
        elif spec.kind == "ratio":
            result = compute_ratio(spec, by_name)
        elif spec.kind == "volume":
            if labels is None:
                raise ValueError(f"volume measurement {spec.name} requires segmentation labels")
            result = compute_labeled_volume(spec, labels, int(raw.get("label_index", raw.get("label", 1))), spacing)
        elif spec.kind == "roi_stat":
            roi_name = spec.roi or str(raw.get("roi", "default"))
            roi_mask = roi_masks.get(roi_name)
            if labels is None or roi_mask is None:
                raise ValueError(f"roi_stat measurement {spec.name} requires labels and ROI mask {roi_name!r}")
            threshold = float(raw.get("threshold", 0.0))
            results.extend(compute_trabecular_metrics(spec, labels > threshold, roi_mask, spacing, threshold))
            by_name.update({item.name: item for item in results})
            continue
        else:
            raise ValueError(f"unsupported measurement kind {spec.kind!r}")
        results.append(result)
        by_name[result.name] = result
        _write_json(output_root / "qc" / f"{result.name}.json", {"measurement": result.name, "inputs": result.inputs})

    workflow_id = _first_str(workflow_measurements, "workflow_id", "unknown-workflow")
    session_id = _first_str(workflow_measurements, "session_id", "unknown-session")
    results_payload = results_to_json(results, workflow_id, session_id)
    qc_payload = build_qc_payload(results)
    overrides_payload = {
        "workflow_id": workflow_id,
        "session_id": session_id,
        "overrides": [asdict(record) for record in _collect_overrides(workflow_measurements, workflow_roi_defs)],
    }

    _write_json(output_root / "results.json", results_payload)
    _write_json(output_root / "qc_overlays.json", qc_payload)
    _write_json(output_root / "overrides.json", overrides_payload)
    (output_root / "summary.md").write_text(results_to_markdown(results), encoding="utf-8")

    return {
        "stage": "measurements",
        "confidence": "high",
        "evidence": f"Computed {len(results)} workflow-defined measurement results with QC payloads.",
        "recommended_action": "proceed",
        "artifacts": {
            "results": str(output_root / "results.json"),
            "qc_overlays": str(output_root / "qc_overlays.json"),
            "overrides": str(output_root / "overrides.json"),
            "summary": str(output_root / "summary.md"),
            "screenshots": [screenshot_path("measurements", 1)],
        },
    }


def _load_landmarks(landmark_artifacts: dict[str, str]) -> dict[str, Any]:
    positions = _load_json(landmark_artifacts.get("positions"))
    if not positions:
        return {}
    if "landmarks" in positions:
        return positions
    return positions


def _load_roi_masks(roi_artifacts: dict[str, Any]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for key, path in roi_artifacts.items():
        if key in {"masks", "roi_masks"}:
            payload = path if isinstance(path, dict) else _load_json(path if Path(path).exists() else None)
            for name, mask_path in payload.items():
                array = _load_roi_mask(str(mask_path))
                if array is not None:
                    masks[str(name)] = array.astype(bool)
        elif key.endswith("_mask"):
            array = _load_roi_mask(path)
            if array is not None:
                masks[key.removesuffix("_mask")] = array.astype(bool)
    return masks


def _load_roi_mask(path: str) -> np.ndarray | None:
    mask_path = Path(path)
    if mask_path.suffix == ".json":
        payload = _load_json(path)
        if "mask_file" in payload:
            linked_path = Path(str(payload["mask_file"]))
            if not linked_path.is_absolute():
                linked_path = mask_path.parent / linked_path
            return _load_array(str(linked_path))
        return np.asarray(payload) if isinstance(payload, list) else None
    return _load_array(path)


def _collect_overrides(measurements: list[dict[str, Any]], rois: list[dict[str, Any]]) -> list[OverrideRecord]:
    records: list[OverrideRecord] = []
    for stage, items in (("measurement", measurements), ("roi", rois)):
        for item in items:
            for raw in item.get("overrides", []) if isinstance(item.get("overrides"), list) else []:
                records.append(_override(stage, raw))
            if "canonical_value" in item and "override_value" in item:
                records.append(_override(stage, item))
    return records


def _override(stage: str, raw: dict[str, Any]) -> OverrideRecord:
    return OverrideRecord(
        stage=str(raw.get("stage", stage)),
        field=str(raw.get("field", "unknown")),
        canonical_value=raw.get("canonical_value"),
        override_value=raw.get("override_value"),
        rationale=str(raw.get("rationale", "run-specific deviation recorded")),
        confidence=str(raw.get("confidence", "medium")),
        approver=str(raw.get("approver", "agent")),
    )


def _load_array(path: str | None) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    if file_path.suffix == ".npz":
        data = np.load(file_path)
        return data[data.files[0]]
    if file_path.suffix == ".json":
        return np.asarray(json.loads(file_path.read_text(encoding="utf-8")))
    if file_path.name.endswith((".nii", ".nii.gz")):
        return np.asarray(nib.load(str(file_path)).get_fdata())
    return None


def _load_json(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _first_str(items: list[dict[str, Any]], field: str, default: str) -> str:
    for item in items:
        value = item.get(field)
        if isinstance(value, str) and value:
            return value
    return default
