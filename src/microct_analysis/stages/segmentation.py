"""Segmentation stage driver — executed via ``jupyter-workbench exec --file``."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import ndimage

from microct_analysis.domain.artifact_contracts import screenshot_path
from microct_analysis.processing import profiles as scanner_profiles
from microct_analysis.processing import resample as resample_processing
from microct_analysis.processing import segmentation as marker_processing
from microct_analysis.processing import threshold as threshold_processing
from microct_analysis.processing import watershed as watershed_processing
from microct_analysis.processing.calibration import (
    DEFAULT_BIMODALITY_RATIO,
    DEFAULT_MARKER_PERCENTILE,
    analyze_histogram,
    analyze_segmentation_histogram,
    derive_segmentation_thresholds,
    derive_thresholds,
)
from microct_analysis.processing.dicom import LoadError, load_dicom
from microct_analysis.processing.io import save_nifti, save_provenance
from microct_analysis.processing.preprocess import median_filter
from microct_analysis.processing.sanity import check_bone_volume_ordering
from microct_analysis.processing.sanity import check as check_segmentation_sanity
from microct_analysis.processing.segmentation import extract_label, seed_from_region, watershed_segment
from microct_analysis.processing.threshold import binary_mask
from microct_analysis.processing.types import (
    BoneLabels,
    Component,
    LabelVolume,
    LoadedScan,
    PrefilteredComponent,
    Provenance,
    ScanFingerprint,
    ScanVolume,
    SeedAssignment,
    Seeds,
    SegmentationResult,
    SegmentationThresholds,
    Thresholds,
)

BONE_ORDER = ("femur", "tibia", "fibula", "patella")
CONFUNDER_LABELS = {
    "sesamoid": "Sesamoid bones near joints may be misidentified as additional structures.",
    "articular-bridging-suspected": "Osteophytes or bridging may connect adjacent bones.",
    "partial-bone-at-scan-boundary": "Partial bones at scan boundaries may bias assignments.",
}

PIPELINE_VERSION = "microct-analysis-segmentation-v1"
ESCALATION_FLAGS = frozenset(
    {
        "histogram-not-bimodal",
        "non-uniform-spacing",
        "anisotropy-too-large",
        "missing-profile",
        "missing-codec",
        "missing-dicom-files",
        "missing-dicom-tags",
        "wrong-modality",
        "ambiguous-bone-identity",
        "articular-bridging-suspected",
        "invalid-seeds",
    }
)
ESCALATION_FLAG_PREFIXES = ("pruning-dropped-bone:",)
RECOVERABLE_FLAGS = frozenset({"ambiguous-bone-identity"})


def run_segmentation(
    volume: Any | None = None,
    spacing: tuple[float, ...] | None = None,
    thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None = None,
    workflow_thresholds: dict[str, Any] | None = None,
    output_dir: str = "segmentation",
    *,
    dicom_path: str | None = None,
    intake_metadata_path: str | None = None,
    scanner: str = "auto",
    scanner_override: str | None = None,
    threshold_method: str = "auto",
    mask_threshold: float | None = None,
    marker_threshold: float | None = None,
    bimodality_ratio: float = DEFAULT_BIMODALITY_RATIO,
    marker_percentile: float = DEFAULT_MARKER_PERCENTILE,
    min_marker_voxels: int = marker_processing.DEFAULT_MIN_MARKER_VOXELS,
    anisotropy_limit: float = 3.0,
    seeds_path: str | None = None,
    fov_prefilter_enabled: bool = True,
    fov_prefilter_margin: int = 3,
    render_qc: bool = False,
) -> dict[str, Any]:
    """Run segmentation from either in-memory inputs or DICOM/intake metadata.

    The original workbench path passes ``volume``/``spacing`` and optionally
    workflow label hints; that path is preserved for backwards compatibility.
    Passing ``dicom_path`` or ``intake_metadata_path`` uses the full processing
    pipeline and emits the richer Step 5 artifact set.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    workflow_thresholds = workflow_thresholds or {}

    if dicom_path is not None or intake_metadata_path is not None:
        report = _run_full_pipeline(
            output_root=output_root,
            dicom_path=dicom_path,
            intake_metadata_path=intake_metadata_path,
            thresholds=thresholds,
            scanner_override=scanner_override or scanner,
            threshold_method=threshold_method,
            mask_threshold=mask_threshold,
            marker_threshold=marker_threshold,
            bimodality_ratio=bimodality_ratio,
            marker_percentile=marker_percentile,
            min_marker_voxels=min_marker_voxels,
            anisotropy_limit=anisotropy_limit,
            seeds_path=Path(seeds_path) if seeds_path is not None else None,
            fov_prefilter_enabled=fov_prefilter_enabled,
            fov_prefilter_margin=fov_prefilter_margin,
            render_qc=render_qc,
        )
        _write_json(output_root / "stage_report.json", report)
        return report

    report = _run_legacy_in_memory(
        volume=volume,
        spacing=spacing,
        thresholds=thresholds,
        workflow_thresholds=workflow_thresholds,
        output_root=output_root,
    )
    _write_json(output_root / "stage_report.json", report)
    return report


def compare_thresholds(derived: Thresholds, workflow_thresholds: dict[str, Any]) -> list[str]:
    """Return discrepancy notes comparing derived and workflow thresholds."""

    observations: list[str] = []
    tolerance = float(workflow_thresholds.get("tolerance_fraction", 0.15))
    fields = {"mask": "bone_soft_tissue", "marker": "subchondral_cortical", "bone_soft_tissue": "bone_soft_tissue"}
    for workflow_field, threshold_field in fields.items():
        expected = _threshold_value(workflow_thresholds, workflow_field)
        if expected is None:
            continue
        actual = float(getattr(derived, threshold_field))
        delta_fraction = abs(actual - float(expected)) / max(abs(float(expected)), 1.0)
        if delta_fraction > tolerance:
            observations.append(
                f"{workflow_field} threshold derived {actual:.3g} differs from workflow {float(expected):.3g} "
                f"by {delta_fraction:.1%} (> {tolerance:.0%})."
            )
    return observations


def detect_confounders(components: list[Any], prefiltered: list[Any], shape: tuple[int, int, int]) -> list[str]:
    """Detect known segmentation confounder signals from component geometry."""

    observations: list[str] = []
    if prefiltered or any(getattr(component, "edge_faces", ()) for component in components):
        observations.append(CONFUNDER_LABELS["partial-bone-at-scan-boundary"])
    if len(components) > 4:
        observations.append(CONFUNDER_LABELS["sesamoid"])
    z_extent = shape[0]
    for component in components:
        bbox = getattr(component, "bbox_zyx", None)
        if bbox is None:
            continue
        (zmin, zmax), _, _ = bbox
        if zmax - zmin > z_extent / 2 and zmin < z_extent / 3 and zmax > 2 * z_extent / 3:
            observations.append(CONFUNDER_LABELS["articular-bridging-suspected"])
            break
    return list(dict.fromkeys(observations))


def confidence_for_segmentation(*, threshold_observations: list[str], confounders: list[str], sanity_warnings: list[str]) -> str:
    if any("articular" in warning or "ambiguous" in warning for warning in sanity_warnings):
        return "low"
    if sanity_warnings or threshold_observations or confounders:
        return "medium"
    return "high"


@dataclasses.dataclass(frozen=True)
class _ComponentInfo:
    index: int
    voxel_count: int
    centroid_zyx: tuple[float, float, float]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    edge_faces: tuple[str, ...]


class _NeedsSeeds(Exception):
    def __init__(self, reason: str, seeds: dict[str, Any], flags: list[str]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.seeds = seeds
        self.flags = flags


def _run_legacy_in_memory(
    *,
    volume: Any | None,
    spacing: tuple[float, ...] | None,
    thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None,
    workflow_thresholds: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Preserve the original volume/spacing workbench behavior."""

    if volume is None or spacing is None:
        return _failure_report(output_root, "segmentation failed: volume and spacing are required without dicom_path")

    try:
        spacing_zyx = _spacing_zyx(spacing)
        scan = _scan_from_inputs(volume, spacing_zyx)
        derived, threshold_flags = _derive_or_fallback_thresholds(scan.data, thresholds)
        threshold_observations = compare_thresholds(derived, workflow_thresholds)
        filtered = median_filter(scan.data)
        mask = binary_mask(filtered, derived.bone_soft_tissue)
        labels, assignments, component_infos = _segment_or_use_labels(filtered, mask, workflow_thresholds, spacing_zyx)
    except _NeedsSeeds as exc:
        _cleanup_ready_artifacts(output_root)
        _write_json(output_root / "seeds.json", exc.seeds)
        _write_json(output_root / "structure_assignments.json", {"status": "needs-seeds", "reason": exc.reason})
        save_provenance({"status": "needs-seeds", "reason": exc.reason, "flags": exc.flags}, output_root / "metadata.json")
        return _report(
            status="needs-seeds",
            confidence="low",
            evidence=f"Structure identification ambiguous; seed curation required. {exc.reason}",
            output_root=output_root,
            flags=exc.flags,
            structure_assignments={},
            threshold_observations=[],
            confounders=[],
        )
    except Exception as exc:  # noqa: BLE001 - stage report captures failures.
        _cleanup_ready_artifacts(output_root)
        return _failure_report(output_root, f"segmentation failed: {exc}")

    confounders = detect_confounders(component_infos, [], scan.data.shape)
    sanity_warnings = check_bone_volume_ordering(labels)
    flags = threshold_flags + (["workflow-threshold-discrepancy"] if threshold_observations else []) + confounders + sanity_warnings

    save_nifti(labels.data, output_root / "labels.nii.gz", scan.affine)
    result = SegmentationResult(labels, assignments, threshold_observations, "high")
    _write_json(
        output_root / "structure_assignments.json",
        {"status": "ready", "assignments": assignments, "flags": flags},
    )
    save_provenance(
        {
            "status": "ready",
            "spacing": spacing_zyx,
            "thresholds": _jsonable(derived),
            "threshold_observations": threshold_observations,
            "structure_assignments": assignments,
            "flags": flags,
            "result": _jsonable(result),
        },
        output_root / "metadata.json",
    )
    _write_json(output_root / "seeds.json", _auto_seed_payload(component_infos))

    confidence = confidence_for_segmentation(
        threshold_observations=threshold_observations,
        confounders=confounders,
        sanity_warnings=sanity_warnings,
    )
    return _report(
        status="ready",
        confidence=confidence,
        evidence=_evidence(confidence, threshold_observations, confounders, sanity_warnings),
        output_root=output_root,
        flags=flags,
        structure_assignments=assignments,
        threshold_observations=threshold_observations,
        confounders=confounders,
    )


def _run_full_pipeline(
    *,
    output_root: Path,
    dicom_path: str | None,
    intake_metadata_path: str | None,
    thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None,
    scanner_override: str,
    threshold_method: str,
    mask_threshold: float | None,
    marker_threshold: float | None,
    bimodality_ratio: float,
    marker_percentile: float,
    min_marker_voxels: int,
    anisotropy_limit: float,
    seeds_path: Path | None,
    fov_prefilter_enabled: bool,
    fov_prefilter_margin: int,
    render_qc: bool,
) -> dict[str, Any]:
    del render_qc  # accepted for workbench/API compatibility; QC rendering lands in a later step.

    flags: list[str] = []
    scan: LoadedScan | None = None
    bone_labels: BoneLabels | None = None
    components: list[Component] = []
    prefiltered: list[PrefilteredComponent] = []
    total_raw_components: int | None = None
    used_seeds: Seeds | None = None
    pruning_stats: dict[str, int] | None = None
    threshold_observations: list[str] = []
    assignments: dict[str, int] = {}
    cc_array: np.ndarray | None = None

    try:
        source_path = _resolve_dicom_path(dicom_path=dicom_path, intake_metadata_path=intake_metadata_path)
        raw_scan = load_dicom(source_path, command=f"segmentation {source_path}", pipeline_version=PIPELINE_VERSION)
        manufacturer = str(raw_scan.provenance.get("manufacturer_raw") or raw_scan.provenance.get("manufacturer") or "")
        model = str(raw_scan.provenance.get("model_raw") or raw_scan.provenance.get("model") or "")
        profile = _resolve_scanner_profile(scanner_override, manufacturer, model)
        if profile.key == "unknown" and scanner_override == "auto" and manufacturer:
            flags.append("unknown-scanner")

        resampled, resampled_spacing = resample_processing.to_isotropic(
            raw_scan.data,
            raw_scan.spacing,
            anisotropy_limit=anisotropy_limit,
        )
        if tuple(float(v) for v in raw_scan.spacing) != tuple(float(v) for v in resampled_spacing):
            flags.append("anisotropic-resampled")
        provenance = _provenance_from_scan(raw_scan, resampled_spacing)
        scan_affine = _affine_for_resampled_scan(raw_scan.affine, raw_scan.spacing, resampled_spacing)

        seg_thresholds, analysis_counts, cal_flags = _derive_segmentation_thresholds_for_stage(
            resampled,
            profile,
            thresholds=thresholds,
            threshold_method=threshold_method,
            mask_threshold=mask_threshold,
            marker_threshold=marker_threshold,
            bimodality_ratio=bimodality_ratio,
            marker_percentile=marker_percentile,
        )
        flags.extend(cal_flags)
        scan = LoadedScan(
            volume=np.asarray(resampled, dtype=np.float32),
            spacing=resampled_spacing,
            affine=scan_affine,
            scanner=profile.key,
            manufacturer_raw=manufacturer,
            model_raw=model,
            thresholds=seg_thresholds,
            provenance=provenance,
            flags=list(flags),
            histogram_sample=analysis_counts,
        )
        _write_full_metadata(
            output_root,
            status=_derive_status(flags),
            flags=flags,
            scan=scan,
            bone_labels=bone_labels,
            components=components,
            prefiltered=prefiltered,
            total_raw_components=total_raw_components,
            seeds=used_seeds,
            pruning_stats=pruning_stats,
        )

        filtered = median_filter(scan.volume, iterations=3, radius=1)
        mask, marker_binary = threshold_processing.apply(filtered, cast(Any, seg_thresholds))
        cc_array, raw_components, total_raw_components = marker_processing.extract_components(
            marker_binary,
            scan.spacing,
            min_marker_voxels,
        )
        components, prefiltered = marker_processing.fov_prefilter(
            raw_components,
            cc_array,
            scan.volume.shape,
            marker_processing.FovPrefilterConfig(enabled=fov_prefilter_enabled, margin_voxels=fov_prefilter_margin),
        )
        fingerprint = _scan_fingerprint(scan, min_marker_voxels, fov_prefilter_enabled, fov_prefilter_margin)

        try:
            if seeds_path is not None:
                used_seeds = _load_seeds(seeds_path)
                assignments = marker_processing.assign_from_seeds(components, used_seeds, fingerprint)
                flags.append("ambiguity-resolved-via-seeds")
                _write_seeds(output_root / "seeds.json", used_seeds)
            else:
                assignments = marker_processing.heuristic_assign(components, scan.volume.shape[0])
        except LoadError as exc:
            if exc.flag != "ambiguous-bone-identity" or seeds_path is not None:
                raise
            flags.append(exc.flag)
            used_seeds = Seeds(
                schema_version=1,
                fingerprint=fingerprint,
                assignments=marker_processing.propose_auto_seeds(components, scan.volume.shape[0]),
            )
            _write_seeds(output_root / "seeds.json", used_seeds)
            _write_json(output_root / "structure_assignments.json", {"status": "needs-seeds", "flags": flags, "assignments": {}})
            if cc_array is not None:
                save_nifti(cc_array.astype(np.int32, copy=False), output_root / "components.nii.gz", scan.affine)
            status = _derive_status(flags)
            _cleanup_ready_artifacts(output_root)
            _write_full_metadata(
                output_root,
                status=status,
                flags=flags,
                scan=scan,
                bone_labels=bone_labels,
                components=components,
                prefiltered=prefiltered,
                total_raw_components=total_raw_components,
                seeds=used_seeds,
                pruning_stats=pruning_stats,
            )
            return _full_report(
                status=status,
                output_root=output_root,
                flags=flags,
                assignments={},
                threshold_observations=threshold_observations,
                components=components,
                prefiltered=prefiltered,
                shape=scan.volume.shape,
                evidence="Structure identification ambiguous; seeds.json emitted for curation.",
            )

        labeling = marker_processing.paint_and_verify(cc_array, assignments, scan.spacing)
        bone_labels = watershed_processing.run(filtered, labeling.labeled_markers, mask, scan.spacing)
        bone_labels, pruning_stats, pruning_flags = watershed_processing.prune_to_seed_cc(
            bone_labels,
            labeling.labeled_markers,
            scan.spacing,
        )
        flags.extend(pruning_flags)
        sanity_flags = check_segmentation_sanity(bone_labels, scan)
        flags.extend(sanity_flags)
        bone_labels = dataclasses.replace(bone_labels, flags=list(flags))
        status = _derive_status(flags)
        label_assignments = _label_assignments(bone_labels) if status == "ready" else {}
        if status != "ready":
            _cleanup_ready_artifacts(output_root)
            if status != "needs-seeds":
                _cleanup_components_artifact(output_root)
        else:
            _cleanup_components_artifact(output_root)
            save_nifti(bone_labels.volume, output_root / "labels.nii.gz", scan.affine)
            _write_bone_masks(output_root / "masks", bone_labels, scan)
        _write_json(
            output_root / "structure_assignments.json",
            {
                "status": status,
                "assignments": label_assignments,
                "component_assignments": assignments if status == "ready" else {},
                "flags": flags,
            },
        )
        if used_seeds is None:
            used_seeds = Seeds(
                schema_version=1,
                fingerprint=fingerprint,
                assignments=marker_processing.propose_auto_seeds(components, scan.volume.shape[0]),
            )
            _write_seeds(output_root / "seeds.json", used_seeds)
        _write_full_metadata(
            output_root,
            status=status,
            flags=flags,
            scan=scan,
            bone_labels=bone_labels if status == "ready" else None,
            components=components,
            prefiltered=prefiltered,
            total_raw_components=total_raw_components,
            seeds=used_seeds,
            pruning_stats=pruning_stats,
        )
        return _full_report(
            status=status,
            output_root=output_root,
            flags=flags,
            assignments=label_assignments,
            threshold_observations=threshold_observations,
            components=components,
            prefiltered=prefiltered,
            shape=scan.volume.shape,
            evidence=_full_evidence(status, flags),
        )
    except LoadError as exc:
        if exc.flag and exc.flag not in flags:
            flags.append(exc.flag)
        _cleanup_ready_artifacts(output_root)
        _cleanup_components_artifact(output_root)
        if used_seeds is None:
            _cleanup_seeds_artifact(output_root)
        status = _derive_status(flags)
        if status == "needs-seeds":
            status = "failed"  # only the explicit ambiguity produce path should emit needs-seeds artifacts.
        _write_json(output_root / "structure_assignments.json", {"status": status, "evidence": str(exc), "flags": flags})
        _write_full_metadata(
            output_root,
            status=status,
            flags=flags,
            scan=scan,
            bone_labels=None,
            components=components,
            prefiltered=prefiltered,
            total_raw_components=total_raw_components,
            seeds=used_seeds,
            pruning_stats=pruning_stats,
        )
        return _full_report(
            status=status,
            output_root=output_root,
            flags=flags,
            assignments={},
            threshold_observations=threshold_observations,
            components=components,
            prefiltered=prefiltered,
            evidence=f"segmentation failed: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - workbench stage report captures failures.
        flags.append("segmentation-failed")
        _cleanup_ready_artifacts(output_root)
        _cleanup_components_artifact(output_root)
        if used_seeds is None:
            _cleanup_seeds_artifact(output_root)
        _write_json(output_root / "structure_assignments.json", {"status": "failed", "evidence": str(exc), "flags": flags})
        _write_full_metadata(
            output_root,
            status="failed",
            flags=flags,
            scan=scan,
            bone_labels=None,
            components=components,
            prefiltered=prefiltered,
            total_raw_components=total_raw_components,
            seeds=used_seeds,
            pruning_stats=pruning_stats,
        )
        return _full_report(
            status="failed",
            output_root=output_root,
            flags=flags,
            assignments={},
            threshold_observations=threshold_observations,
            components=components,
            prefiltered=prefiltered,
            evidence=f"segmentation failed: {exc}",
        )


def _derive_or_fallback_thresholds(volume: np.ndarray, thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None) -> tuple[Thresholds, list[str]]:
    manual = _thresholds_from_input(thresholds)
    if manual is not None:
        return manual, ["manual-thresholds-used"]
    scanner = thresholds.get("scanner", "scanco") if isinstance(thresholds, dict) else "scanco"
    _ = analyze_histogram(volume)
    return derive_thresholds(volume, scanner=str(scanner)), []


def _segment_or_use_labels(
    filtered: np.ndarray, mask: np.ndarray, workflow_thresholds: dict[str, Any], spacing: tuple[float, float, float]
) -> tuple[LabelVolume, dict[str, int], list[_ComponentInfo]]:
    labeled_input = workflow_thresholds.get("labels") if "labels" in workflow_thresholds else workflow_thresholds.get("labeled_input")
    if labeled_input is not None:
        data = np.asarray(labeled_input, dtype=np.uint16)
        assignments = _assign_from_labels(data)
        if len(assignments) < 2:
            raise _NeedsSeeds("labeled input contains too few structures", {}, ["ambiguous-bone-identity"])
        return LabelVolume(data=data, spacing=spacing, label_map=assignments), assignments, _components_from_labels(data)

    components = _components_from_mask(mask)
    if len(components) < 2:
        raise _NeedsSeeds("fewer than two bone-like components found", _auto_seed_payload(components), ["ambiguous-bone-identity"])
    assignments = {name: component.index for name, component in zip(BONE_ORDER, components, strict=False)}
    seed_points = {
        name: tuple(int(round(v)) for v in component.centroid_zyx) for name, component in zip(assignments, components, strict=False)
    }
    seeds = seed_from_region(filtered, mask, seed_points)
    grown = watershed_segment(filtered, seeds, mask).astype(np.uint16)
    # Exercise extract_label and normalize labels to assignment IDs.
    normalized = np.zeros_like(grown, dtype=np.uint16)
    for label_id in assignments.values():
        normalized[extract_label(grown, label_id)] = label_id
    return LabelVolume(data=normalized, spacing=spacing, label_map=assignments), assignments, components


def _components_from_mask(mask: np.ndarray) -> list[_ComponentInfo]:
    labeled, count = cast(tuple[np.ndarray, int], ndimage.label(mask))
    components = [_component_info(labeled, index) for index in range(1, count + 1)]
    return sorted(components, key=lambda item: item.voxel_count, reverse=True)


def _components_from_labels(labels: np.ndarray) -> list[_ComponentInfo]:
    return sorted(
        (_component_info(labels == label_id, int(label_id)) for label_id in np.unique(labels) if int(label_id) != 0),
        key=lambda item: item.voxel_count,
        reverse=True,
    )


def _component_info(labeled: np.ndarray, index: int) -> _ComponentInfo:
    coords = np.argwhere(labeled == index) if labeled.dtype.kind in "iu" else np.argwhere(labeled)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    shape = labeled.shape
    edge_faces = []
    for axis, name in enumerate("zyx"):
        if mins[axis] == 0:
            edge_faces.append(f"{name}_min")
        if maxs[axis] == shape[axis]:
            edge_faces.append(f"{name}_max")
    return _ComponentInfo(
        index=index,
        voxel_count=int(coords.shape[0]),
        centroid_zyx=tuple(float(v) for v in coords.mean(axis=0)),  # type: ignore[return-value]
        bbox_zyx=tuple((int(mins[i]), int(maxs[i])) for i in range(3)),  # type: ignore[assignment]
        edge_faces=tuple(edge_faces),
    )


def _assign_from_labels(labels: np.ndarray) -> dict[str, int]:
    components = _components_from_labels(labels)
    return {name: component.index for name, component in zip(BONE_ORDER, components, strict=False)}


def _auto_seed_payload(components: list[Any]) -> dict[str, Any]:
    return {
        "status": "auto-proposed",
        "seeds": [
            {"component_index": c.index, "voxel_count": c.voxel_count, "centroid_zyx": list(c.centroid_zyx)} for c in components
        ],
    }


def _scan_from_inputs(volume: Any, spacing: tuple[float, float, float]) -> ScanVolume:
    array = np.asarray(volume, dtype=np.float32)
    affine = _affine_from_spacing(spacing)
    return ScanVolume(data=array, spacing=spacing, affine=affine, provenance={"source": "workbench-session"})


def _resolve_dicom_path(*, dicom_path: str | None, intake_metadata_path: str | None) -> Path:
    if dicom_path is not None:
        return Path(dicom_path)
    if intake_metadata_path is None:
        raise LoadError("missing-dicom-files", "dicom_path or intake_metadata_path is required")
    metadata_file = Path(intake_metadata_path)
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LoadError("missing-dicom-files", f"intake metadata not found: {metadata_file}") from exc
    except json.JSONDecodeError as exc:
        raise LoadError("missing-dicom-tags", f"intake metadata is not valid JSON: {exc}") from exc
    candidate = metadata.get("dicom_path")
    provenance = metadata.get("provenance")
    if candidate is None and isinstance(provenance, dict):
        candidate = provenance.get("source_dir")
    if not candidate:
        raise LoadError("missing-dicom-files", "intake metadata does not include dicom_path/provenance.source_dir")
    resolved = Path(str(candidate))
    if not resolved.is_absolute():
        resolved = metadata_file.parent / resolved
    return resolved


def _resolve_scanner_profile(scanner_override: str, manufacturer: str, model: str) -> scanner_profiles.ScannerProfile:
    key = scanner_override.lower()
    if key == "auto":
        return scanner_profiles.detect(manufacturer, model)
    if key in {"scanco", "unknown"}:
        return scanner_profiles.get(key)
    raise LoadError("missing-profile", f"unknown scanner override: {scanner_override!r}")


def _derive_segmentation_thresholds_for_stage(
    volume: np.ndarray,
    profile: scanner_profiles.ScannerProfile,
    *,
    thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None,
    threshold_method: str,
    mask_threshold: float | None,
    marker_threshold: float | None,
    bimodality_ratio: float,
    marker_percentile: float,
) -> tuple[SegmentationThresholds, list[int], list[str]]:
    manual = _segmentation_thresholds_from_input(thresholds, mask_threshold, marker_threshold)
    analysis = analyze_segmentation_histogram(volume, bimodality_ratio=bimodality_ratio)
    if manual is not None:
        return manual, analysis.counts.tolist(), ["manual-thresholds-used"]

    method = threshold_method.lower()
    if method == "profile":
        if not profile.has_documented_thresholds or profile.profile_mask_threshold is None or profile.profile_marker_threshold is None:
            raise LoadError("missing-profile", f"profile {profile.key!r} has no documented segmentation thresholds")
        return (
            SegmentationThresholds(
                mask=float(profile.profile_mask_threshold),
                marker=float(profile.profile_marker_threshold),
                method="scanner-profile",
            ),
            analysis.counts.tolist(),
            [],
        )
    if method == "histogram":
        derived, histogram, flags = derive_segmentation_thresholds(
            volume,
            scanner_profiles.UNKNOWN,
            bimodality_ratio=bimodality_ratio,
            marker_percentile=marker_percentile,
        )
        return derived, histogram.counts.tolist(), flags
    if method == "auto":
        derived, histogram, flags = derive_segmentation_thresholds(
            volume,
            profile,
            bimodality_ratio=bimodality_ratio,
            marker_percentile=marker_percentile,
        )
        return derived, histogram.counts.tolist(), flags
    raise LoadError("histogram-not-bimodal", f"unknown threshold method: {threshold_method!r}")


def _segmentation_thresholds_from_input(
    thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None,
    mask_threshold: float | None,
    marker_threshold: float | None,
) -> SegmentationThresholds | None:
    mask = mask_threshold
    marker = marker_threshold
    if isinstance(thresholds, SegmentationThresholds):
        return thresholds
    if isinstance(thresholds, Thresholds):
        mask = thresholds.bone_soft_tissue if mask is None else mask
        marker = thresholds.subchondral_cortical if marker is None else marker
    elif isinstance(thresholds, dict):
        if mask is None:
            mask = _first_threshold_value(thresholds, "mask", "bone_soft_tissue")
        if marker is None:
            marker = _first_threshold_value(thresholds, "marker", "subchondral_cortical")
    if mask is None and marker is None:
        return None
    if mask is None or marker is None:
        raise LoadError("histogram-not-bimodal", "manual mask and marker thresholds must be supplied together")
    return SegmentationThresholds(mask=float(mask), marker=float(marker), method="manual-override")


def _affine_for_resampled_scan(
    source_affine: np.ndarray,
    source_spacing: tuple[float, ...],
    output_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Preserve DICOM orientation/origin while updating voxel spacing."""

    affine = np.asarray(source_affine, dtype=np.float64).copy()
    if affine.shape != (4, 4):
        return _affine_from_spacing(output_spacing)
    source_spacing_zyx = _spacing_zyx(tuple(float(value) for value in source_spacing))
    source_spacing_xyz = (source_spacing_zyx[2], source_spacing_zyx[1], source_spacing_zyx[0])
    output_spacing_xyz = (output_spacing[2], output_spacing[1], output_spacing[0])
    for axis, (old_spacing, new_spacing) in enumerate(zip(source_spacing_xyz, output_spacing_xyz, strict=True)):
        column = affine[:3, axis]
        norm = float(np.linalg.norm(column))
        if norm > 0:
            affine[:3, axis] = column / norm * new_spacing
        else:
            affine[axis, axis] = new_spacing
    return affine


def _provenance_from_scan(scan: ScanVolume, resampled_spacing: tuple[float, float, float]) -> Provenance:
    provenance = scan.provenance
    original_spacing = tuple(float(v) for v in provenance.get("original_spacing", scan.spacing))
    source_dir = str(provenance.get("source_dir") or provenance.get("dicom_path") or "")
    resampled_value = list(resampled_spacing) if tuple(original_spacing) != tuple(float(v) for v in resampled_spacing) else None
    return Provenance(
        source_dir=source_dir,
        n_slices=int(provenance.get("n_slices") or provenance.get("slice_count") or scan.data.shape[0]),
        original_spacing=cast(tuple[float, float, float], original_spacing),
        resampled_spacing=cast(tuple[float, float, float] | None, tuple(resampled_value) if resampled_value is not None else None),
        slice_uid_hash=str(provenance.get("slice_uid_hash") or "workbench-session"),
        pipeline_version=PIPELINE_VERSION,
        command=str(provenance.get("command") or "segmentation"),
        manufacturer_raw=str(provenance.get("manufacturer_raw") or provenance.get("manufacturer") or ""),
        model_raw=str(provenance.get("model_raw") or provenance.get("model") or ""),
        transfer_syntax_uid=str(provenance.get("transfer_syntax_uid") or ""),
    )


def _scan_fingerprint(scan: LoadedScan, min_marker_voxels: int, fov_enabled: bool, fov_margin: int) -> ScanFingerprint:
    return ScanFingerprint(
        slice_uid_hash=scan.provenance.slice_uid_hash,
        volume_shape=_tuple3_int(scan.volume.shape, "volume_shape"),
        mask_threshold=float(scan.thresholds.mask),
        marker_threshold=float(scan.thresholds.marker),
        min_marker_voxels=int(min_marker_voxels),
        fov_prefilter_enabled=bool(fov_enabled),
        fov_prefilter_margin=int(fov_margin),
    )


def _tuple3_int(values: Any, field: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{field} must contain three values")
    return (result[0], result[1], result[2])


def _tuple3_float(values: Any, field: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{field} must contain three values")
    return (result[0], result[1], result[2])


def _write_full_metadata(
    output_root: Path,
    *,
    status: str,
    flags: list[str],
    scan: LoadedScan | None,
    bone_labels: BoneLabels | None,
    components: list[Component],
    prefiltered: list[PrefilteredComponent],
    total_raw_components: int | None,
    seeds: Seeds | None,
    pruning_stats: dict[str, int] | None,
) -> None:
    payload: dict[str, Any] = {"status": status, "flags": list(flags)}
    if scan is not None:
        payload.update(
            {
                "scanner": scan.scanner,
                "manufacturer_raw": scan.manufacturer_raw,
                "model_raw": scan.model_raw,
                "spacing": list(scan.spacing),
                "spacing_zyx_mm": list(scan.spacing),
                "thresholds": _jsonable(scan.thresholds),
                "provenance": _jsonable(scan.provenance),
            }
        )
        if scan.histogram_sample is not None:
            payload["histogram_sample"] = scan.histogram_sample
    if components or prefiltered or total_raw_components is not None:
        payload["components"] = _components_block(components, prefiltered, total_raw_components)
    if seeds is not None:
        payload["seeds"] = _seeds_to_dict(seeds)
    if bone_labels is not None:
        payload["per_bone"] = {name: _jsonable(stats) for name, stats in bone_labels.per_bone.items()}
    if pruning_stats is not None:
        payload["pruning_stats"] = {name: int(count) for name, count in pruning_stats.items()}
        payload["post_watershed_pruning"] = {"voxels_removed_per_bone": payload["pruning_stats"]}
    save_provenance(payload, output_root / "metadata.json")


def _components_block(
    components: list[Component],
    prefiltered: list[PrefilteredComponent],
    total_raw_components: int | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if total_raw_components is not None:
        block["total_raw"] = int(total_raw_components)
    block["above_min_voxels"] = len(components) + len(prefiltered)
    block["retained"] = len(components)
    if components:
        sizes = np.array([component.voxel_count for component in components], dtype=np.int64)
        block["size_quantiles"] = {
            "p50": int(np.percentile(sizes, 50)),
            "p90": int(np.percentile(sizes, 90)),
            "p99": int(np.percentile(sizes, 99)),
            "max": int(sizes.max()),
        }
        edge_counts = {key: 0 for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
        for component in components:
            for face in component.edge_faces:
                if face in edge_counts:
                    edge_counts[face] += 1
        block["edge_touch_counts"] = edge_counts
        block["top"] = [_component_to_dict(component) for component in sorted(components, key=lambda item: item.voxel_count, reverse=True)[:50]]
    else:
        block["size_quantiles"] = None
        block["edge_touch_counts"] = None
        block["top"] = []
    block["prefiltered"] = [
        {**_component_to_dict(item.component), "reasons": list(item.reasons)} for item in prefiltered
    ]
    return block


def _component_to_dict(component: Component) -> dict[str, Any]:
    return {
        "index": component.index,
        "voxel_count": component.voxel_count,
        "centroid_zyx": list(component.centroid_zyx),
        "centroid_mm": list(component.centroid_mm),
        "bbox_zyx": [list(axis) for axis in component.bbox_zyx],
        "edge_faces": list(component.edge_faces),
    }


def _load_seeds(path: Path) -> Seeds:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LoadError("invalid-seeds", f"seeds file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LoadError("invalid-seeds", f"seeds file not valid JSON: {exc}") from exc
    try:
        fp_data = data["scan_fingerprint"]
        fov = fp_data["fov_prefilter"]
        fingerprint = ScanFingerprint(
            slice_uid_hash=str(fp_data["slice_uid_hash"]),
            volume_shape=_tuple3_int(fp_data["volume_shape"], "volume_shape"),
            mask_threshold=float(fp_data["mask_threshold"]),
            marker_threshold=float(fp_data["marker_threshold"]),
            min_marker_voxels=int(fp_data["min_marker_voxels"]),
            fov_prefilter_enabled=bool(fov["enabled"]),
            fov_prefilter_margin=int(fov["margin_voxels"]),
        )
        assignments = {
            str(name): SeedAssignment(
                component_index=int(assignment["component_index"]),
                voxel_count=int(assignment["voxel_count"]),
                centroid_zyx=_tuple3_float(assignment["centroid_zyx"], "centroid_zyx"),
            )
            for name, assignment in data["assignments"].items()
        }
        schema_version = int(data.get("schema_version", 1))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise LoadError("invalid-seeds", f"seeds schema malformed: {exc}") from exc
    return Seeds(schema_version=schema_version, fingerprint=fingerprint, assignments=assignments)


def _write_seeds(path: Path, seeds: Seeds) -> None:
    _write_json(path, _seeds_to_dict(seeds))


def _seeds_to_dict(seeds: Seeds) -> dict[str, Any]:
    fingerprint = seeds.fingerprint
    return {
        "schema_version": seeds.schema_version,
        "scan_fingerprint": {
            "slice_uid_hash": fingerprint.slice_uid_hash,
            "volume_shape": list(fingerprint.volume_shape),
            "mask_threshold": fingerprint.mask_threshold,
            "marker_threshold": fingerprint.marker_threshold,
            "min_marker_voxels": fingerprint.min_marker_voxels,
            "fov_prefilter": {
                "enabled": fingerprint.fov_prefilter_enabled,
                "margin_voxels": fingerprint.fov_prefilter_margin,
            },
        },
        "assignments": {
            name: {
                "component_index": assignment.component_index,
                "voxel_count": assignment.voxel_count,
                "centroid_zyx": list(assignment.centroid_zyx),
            }
            for name, assignment in seeds.assignments.items()
        },
    }


def _label_assignments(bone_labels: BoneLabels) -> dict[str, int]:
    return {name: stats.label for name, stats in bone_labels.per_bone.items()}


def _write_bone_masks(masks_dir: Path, bone_labels: BoneLabels, scan: LoadedScan) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    for name, stats in bone_labels.per_bone.items():
        mask = (bone_labels.volume == stats.label).astype(np.uint8)
        save_nifti(mask, masks_dir / f"{name}.nii.gz", scan.affine)


def _cleanup_ready_artifacts(output_root: Path) -> None:
    labels_path = output_root / "labels.nii.gz"
    if labels_path.exists():
        labels_path.unlink()
    masks_dir = output_root / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)


def _cleanup_components_artifact(output_root: Path) -> None:
    components_path = output_root / "components.nii.gz"
    if components_path.exists():
        components_path.unlink()


def _cleanup_seeds_artifact(output_root: Path) -> None:
    seeds_path = output_root / "seeds.json"
    if seeds_path.exists():
        seeds_path.unlink()


def _derive_status(flags: list[str]) -> str:
    escalating = [flag for flag in flags if _is_escalation(flag)]
    if not escalating:
        return "ready"
    if any(flag not in RECOVERABLE_FLAGS for flag in escalating):
        return "failed"
    return "needs-seeds"


def _is_escalation(flag: str) -> bool:
    return flag in ESCALATION_FLAGS or flag.startswith(ESCALATION_FLAG_PREFIXES)


def _full_report(
    *,
    status: str,
    output_root: Path,
    flags: list[str],
    assignments: dict[str, int],
    threshold_observations: list[str],
    components: list[Component],
    prefiltered: list[PrefilteredComponent],
    evidence: str,
    shape: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, Any]:
    confounders = detect_confounders(components, prefiltered, shape) if components or prefiltered else []
    confidence = "low" if status != "ready" else "medium" if flags or confounders or threshold_observations else "high"
    return _report(
        status=status,
        confidence=confidence,
        evidence=evidence,
        output_root=output_root,
        flags=flags,
        structure_assignments=assignments,
        threshold_observations=threshold_observations,
        confounders=confounders,
    )


def _full_evidence(status: str, flags: list[str]) -> str:
    if status == "ready" and not flags:
        return "Segmentation completed; thresholds, structure assignments, and sanity checks agree."
    if status == "ready":
        return "Segmentation completed with observations: " + ", ".join(flags)
    if status == "needs-seeds":
        return "Structure identification ambiguous; seed curation required."
    return "Segmentation failed with escalation flags: " + ", ".join(flags)


def _thresholds_from_input(thresholds: dict[str, Any] | Thresholds | SegmentationThresholds | None) -> Thresholds | None:
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, SegmentationThresholds):
        return Thresholds(bone_soft_tissue=int(thresholds.mask), subchondral_cortical=int(thresholds.marker))
    if not isinstance(thresholds, dict):
        return None
    bone = _first_threshold_value(thresholds, "bone_soft_tissue", "mask")
    cortical = _first_threshold_value(thresholds, "subchondral_cortical", "marker")
    if bone is None or cortical is None:
        return None
    return Thresholds(bone_soft_tissue=int(bone), subchondral_cortical=int(cortical))


def _first_threshold_value(values: dict[str, Any], *fields: str) -> float | None:
    for field in fields:
        value = _threshold_value(values, field)
        if value is not None:
            return value
    return None


def _threshold_value(values: dict[str, Any], field: str) -> float | None:
    for key in (field, f"{field}_threshold"):
        if key in values and not isinstance(values[key], dict):
            return float(values[key])
    nested = values.get(field)
    if isinstance(nested, dict):
        for key in ("value", "threshold"):
            if key in nested:
                return float(nested[key])
    return None


def _spacing_zyx(spacing: tuple[float, ...]) -> tuple[float, float, float]:
    if len(spacing) != 3:
        raise ValueError("spacing must contain z, y, x values")
    spacing_zyx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    if any(value <= 0 for value in spacing_zyx):
        raise ValueError("spacing values must be positive")
    return spacing_zyx


def _affine_from_spacing(spacing: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing[2], spacing[1], spacing[0]
    return affine


def _report(*, status: str, confidence: str, evidence: str, output_root: Path, flags: list[str], structure_assignments: dict[str, int], threshold_observations: list[str], confounders: list[str]) -> dict[str, Any]:
    recommended_action = "pause" if confidence == "low" else "flag" if confidence == "medium" else "proceed"
    return {
        "stage": "segmentation",
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": recommended_action,
        "artifacts": {
            "labels": str(output_root / "labels.nii.gz"),
            "structure_assignments": str(output_root / "structure_assignments.json"),
            "seeds": str(output_root / "seeds.json"),
            "screenshots": [screenshot_path("segmentation", 1)],
        },
        "structure_assignments": structure_assignments,
        "threshold_observations": threshold_observations,
        "confounders": confounders,
        "flags": flags,
    }


def _failure_report(output_root: Path, evidence: str) -> dict[str, Any]:
    _write_json(output_root / "structure_assignments.json", {"status": "failed", "evidence": evidence})
    save_provenance({"status": "failed", "flags": ["segmentation-failed"], "evidence": evidence}, output_root / "metadata.json")
    return _report(status="failed", confidence="low", evidence=evidence, output_root=output_root, flags=["segmentation-failed"], structure_assignments={}, threshold_observations=[], confounders=[])


def _evidence(confidence: str, threshold_observations: list[str], confounders: list[str], sanity_warnings: list[str]) -> str:
    if confidence == "high":
        return "Segmentation completed; thresholds, structure assignments, and sanity checks agree."
    parts = ["Segmentation completed with observations."]
    parts.extend(threshold_observations)
    parts.extend(confounders)
    parts.extend(sanity_warnings)
    return " ".join(parts)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
