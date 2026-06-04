"""Artifact path contracts and stage report structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntakeArtifacts:
    volume_metadata: str = "intake/volume_metadata.json"
    orientation_report: str = "intake/orientation_report.md"
    stage_report: str = "intake/stage_report.json"
    screenshot: str = "intake/screenshot_001.png"


@dataclass(frozen=True)
class SegmentationArtifacts:
    labels: str = "segmentation/labels.nii.gz"
    filtered: str = "segmentation/filtered.nii.gz"
    structure_assignments: str = "segmentation/structure_assignments.json"
    seeds: str = "segmentation/seeds.json"
    metadata: str = "segmentation/metadata.json"
    stage_report: str = "segmentation/stage_report.json"
    components: str = "segmentation/components.nii.gz"
    masks_dir: str = "segmentation/masks"


@dataclass(frozen=True)
class LandmarkArtifacts:
    positions: str = "landmarks/positions.json"
    orientation_frame: str = "landmarks/orientation_frame.json"


@dataclass(frozen=True)
class OrientationArtifacts:
    transform_matrix: str = "landmarks/transform_matrix.json"
    oriented_labels: str = "landmarks/oriented_labels.npy"
    orientation_report: str = "landmarks/orientation_report.md"


@dataclass(frozen=True)
class MeasurementArtifacts:
    results: str = "measurements/results.json"
    qc_overlays: str = "measurements/qc_overlays.json"
    overrides: str = "measurements/overrides.json"


@dataclass(frozen=True)
class StageReport:
    """Structured report from a specialist sub-agent."""

    stage: str
    confidence: str
    evidence: str
    recommended_action: str
    artifacts: dict[str, str | list[str]]
    screenshots: list[str] = field(default_factory=list)


def screenshot_path(stage: str, index: int) -> str:
    """Return ``<stage>/screenshot_<NNN>.png`` for a one-based index."""

    if not stage or "/" in stage or "\\" in stage:
        raise ValueError("stage must be a non-empty single path component")
    if index < 1:
        raise ValueError("index must be >= 1")
    return f"{stage}/screenshot_{index:03d}.png"
