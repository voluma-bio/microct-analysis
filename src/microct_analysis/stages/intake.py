"""Intake stage driver — executed via jupyter-workbench exec --file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microct_analysis.domain.artifact_contracts import IntakeArtifacts, screenshot_path
from microct_analysis.processing import profiles
from microct_analysis.processing.calibration import (
    analyze_histogram,
    analyze_segmentation_histogram,
    derive_segmentation_thresholds,
    derive_thresholds,
)
from microct_analysis.processing.dicom import LoadError, load_dicom
from microct_analysis.processing.types import ScanVolume, SegmentationThresholds

PIPELINE_VERSION = "microct-analysis-intake-v1"


def _json_ready(value: Any) -> Any:
    """Convert common dataclass, array, and scalar objects to JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return {field: _json_ready(getattr(value, field)) for field in value.__dataclass_fields__}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_intake(dicom_path: str, output_dir: str = ".") -> dict[str, Any]:
    """Load and validate a DICOM scan, then write intake artifacts."""

    artifacts = IntakeArtifacts()
    root = Path(output_dir)
    (root / "intake").mkdir(parents=True, exist_ok=True)

    try:
        volume: ScanVolume = load_dicom(Path(dicom_path))
    except LoadError:
        raise

    provenance = dict(volume.provenance)
    provenance["command"] = f"intake {dicom_path} --output-dir {output_dir}"
    provenance["pipeline_version"] = PIPELINE_VERSION
    manufacturer = str(provenance.get("manufacturer") or provenance.get("manufacturer_raw") or "")
    model = str(provenance.get("model") or provenance.get("model_raw") or "")
    scanner_profile = profiles.detect(manufacturer, model)
    scanner = scanner_profile.key
    source_dir = str(provenance.get("source_dir") or Path(dicom_path).resolve())

    histogram = analyze_histogram(volume.data)
    thresholds = derive_thresholds(volume.data, scanner=scanner)
    threshold_summary, segmentation_thresholds, threshold_flags = _segmentation_threshold_summary(volume, scanner_profile)

    metadata: dict[str, Any] = {
        "dicom_path": str(dicom_path),
        "dicom_source_dir": source_dir,
        "source_dir": source_dir,
        "spacing": _json_ready(volume.spacing),
        "original_spacing": _json_ready(list(volume.spacing)),
        "affine": _json_ready(volume.affine),
        "manufacturer": manufacturer,
        "model": model,
        "voxel_count": int(volume.data.size),
        "provenance": _json_ready(provenance),
        "fingerprint": _fingerprint_from_provenance(provenance, source_dir),
        "scanner": scanner,
        "scanner_profile": scanner,
        "scanner_detection": _scanner_detection(scanner_profile, manufacturer, model),
        "histogram": _json_ready(histogram),
        "threshold_analysis": _json_ready(histogram),
        "segmentation_threshold_analysis": threshold_summary,
        "segmentation_ready": segmentation_thresholds is not None,
        "derived_thresholds": _json_ready(thresholds),
        "segmentation_thresholds": _json_ready(segmentation_thresholds),
        "threshold_flags": threshold_flags,
        "scan_volume_contract": ScanVolume.__name__,
        "loaded_scan_contract": ScanVolume.__name__,
    }

    metadata_path = root / artifacts.volume_metadata
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    orientation_report = (
        "# Intake orientation report\n\n"
        "Loaded scan through local `microct_analysis.processing.dicom.load_dicom` and derived "
        "initial thresholds through local calibration helpers.\n\n"
        f"- DICOM input: `{dicom_path}`\n"
        f"- DICOM source dir: `{source_dir}`\n"
        f"- Spacing: `{metadata['spacing']}`\n"
        f"- Manufacturer: `{metadata['manufacturer']}`\n"
        f"- Model: `{metadata['model']}`\n"
        f"- Scanner profile: `{metadata['scanner_profile']}`\n"
        f"- Slice UID hash: `{metadata['fingerprint']['slice_uid_hash']}`\n"
        "- Resampling: deferred to segmentation; no raw/resampled volume is emitted by intake.\n"
    )
    orientation_path = root / artifacts.orientation_report
    orientation_path.write_text(orientation_report, encoding="utf-8")

    stage_report = _stage_report(root, artifacts, metadata, threshold_flags)
    (root / artifacts.stage_report).write_text(json.dumps(stage_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return metadata


def _segmentation_threshold_summary(
    volume: ScanVolume,
    scanner_profile: profiles.ScannerProfile,
) -> tuple[dict[str, Any], SegmentationThresholds | None, list[str]]:
    analysis = analyze_segmentation_histogram(volume.data)
    summary: dict[str, Any] = {
        "status": "analyzed" if analysis.is_bimodal else "not-bimodal",
        "profile": scanner_profile.key,
        "otsu_threshold": analysis.otsu_threshold,
        "separability": analysis.separability,
        "is_bimodal": analysis.is_bimodal,
        "histogram_bins": int(len(analysis.counts)),
        "histogram_range": [float(analysis.bin_edges[0]), float(analysis.bin_edges[-1])],
        "histogram_sample": analysis.counts.tolist(),
    }
    flags: list[str] = []
    if scanner_profile.key == "unknown":
        flags.append("unknown-scanner-profile")
    try:
        thresholds, derived_analysis, calibration_flags = derive_segmentation_thresholds(volume.data, scanner_profile)
    except LoadError as exc:
        if exc.flag not in flags:
            flags.append(exc.flag)
        summary["status"] = "failed"
        summary["error_flag"] = exc.flag
        summary["error"] = str(exc)
        return _json_ready(summary), None, flags

    summary.update(
        {
            "status": "derived",
            "otsu_threshold": derived_analysis.otsu_threshold,
            "separability": derived_analysis.separability,
            "is_bimodal": derived_analysis.is_bimodal,
            "thresholds": _json_ready(thresholds),
        }
    )
    for flag in calibration_flags:
        if flag not in flags:
            flags.append(flag)
    return _json_ready(summary), thresholds, flags


def _fingerprint_from_provenance(provenance: dict[str, Any], source_dir: str) -> dict[str, Any]:
    return {
        "slice_uid_hash": str(provenance.get("slice_uid_hash") or ""),
        "slice_count": int(provenance.get("slice_count") or provenance.get("n_slices") or 0),
        "source_dir": source_dir,
        "transfer_syntax_uid": str(provenance.get("transfer_syntax_uid") or ""),
    }


def _scanner_detection(profile: profiles.ScannerProfile, manufacturer: str, model: str) -> dict[str, Any]:
    return {
        "profile": profile.key,
        "manufacturer": manufacturer,
        "model": model,
        "matched": profile.key != "unknown",
        "has_documented_thresholds": profile.has_documented_thresholds,
    }


def _stage_report(root: Path, artifacts: IntakeArtifacts, metadata: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    confidence = "medium" if flags else "high"
    recommended_action = "flag" if flags else "proceed"
    source_dir = str(metadata.get("dicom_source_dir") or metadata.get("dicom_path") or "")
    evidence = (
        f"Loaded {metadata.get('provenance', {}).get('slice_count', 0)} CT slices from {source_dir}; "
        f"scanner profile {metadata.get('scanner_profile', 'unknown')}."
    )
    if flags:
        evidence += " Observations: " + ", ".join(flags) + "."
    threshold_status = "unknown"
    threshold_analysis = metadata.get("segmentation_threshold_analysis")
    if isinstance(threshold_analysis, dict):
        threshold_status = str(threshold_analysis.get("status", "unknown"))
    return {
        "stage": "intake",
        "status": "ready",
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": recommended_action,
        "artifacts": {
            "volume_metadata": str(root / artifacts.volume_metadata),
            "orientation_report": str(root / artifacts.orientation_report),
            "stage_report": str(root / artifacts.stage_report),
            "screenshots": [screenshot_path("intake", 1)],
        },
        "flags": list(flags),
        "scanner_profile": metadata.get("scanner_profile", "unknown"),
        "spacing": metadata.get("spacing", []),
        "segmentation_ready": bool(metadata.get("segmentation_ready", False)),
        "threshold_analysis_status": threshold_status,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run microCT intake stage")
    parser.add_argument("dicom_path")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    print(json.dumps(run_intake(args.dicom_path, args.output_dir), indent=2, sort_keys=True))
