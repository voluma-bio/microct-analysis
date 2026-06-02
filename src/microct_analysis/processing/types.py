"""Core processing-layer data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Thresholds:
    """SCANCO-equivalent processing thresholds used by legacy workflow code."""

    bone_soft_tissue: int = 220
    subchondral_cortical: int = 270
    surface_3d: int = 320
    marrow_bone: int = 270


@dataclass(frozen=True, slots=True)
class SegmentationThresholds:
    """Mask/marker thresholds used by the end-to-end segmentation pipeline."""

    mask: float
    marker: float
    method: str


@dataclass(frozen=True)
class ScanVolume:
    """Loaded scan data with physical spacing and provenance."""

    data: NDArray[np.float32]
    spacing: tuple[float, ...]
    affine: NDArray[np.float64]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Provenance:
    """DICOM/intake provenance that can fingerprint a segmentation rerun."""

    source_dir: str
    n_slices: int
    original_spacing: tuple[float, float, float]
    resampled_spacing: tuple[float, float, float] | None
    slice_uid_hash: str
    pipeline_version: str
    command: str
    manufacturer_raw: str
    model_raw: str
    transfer_syntax_uid: str


@dataclass(frozen=True, slots=True)
class LoadedScan:
    """Loaded/resampled scan state for the segmentation pipeline."""

    volume: NDArray[np.float32]
    spacing: tuple[float, float, float]
    affine: NDArray[np.float64]
    scanner: str
    manufacturer_raw: str
    model_raw: str
    thresholds: SegmentationThresholds
    provenance: Provenance
    flags: list[str] = field(default_factory=list)
    histogram_sample: list[int] | None = None


@dataclass(frozen=True)
class LabelVolume:
    """Discrete labels with 0 reserved for background."""

    data: NDArray[np.uint8] | NDArray[np.uint16]
    spacing: tuple[float, ...]
    label_map: dict[str, int]


@dataclass(frozen=True)
class SegmentationResult:
    """Result of assigning segmented structures to labels."""

    labels: LabelVolume
    structure_assignments: dict[str, int]
    threshold_observations: list[str]
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class BoneStats:
    label: int
    name: str
    voxel_count: int
    volume_mm3: float
    centroid_mm: tuple[float, float, float]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class BoneLabels:
    """Canonical labeled bone volume. 0=background, 1+=bones/other."""

    volume: NDArray[np.uint8]
    per_bone: dict[str, BoneStats]
    flags: list[str] = field(default_factory=list)


BONE_LABELS: dict[int, str] = {
    1: "femur",
    2: "tibia",
    3: "fibula",
    4: "patella",
}

REQUIRED_SEED_BONES: frozenset[str] = frozenset({"femur", "tibia"})
OPTIONAL_SEED_BONES: frozenset[str] = frozenset({"fibula", "patella"})


@dataclass(frozen=True, slots=True)
class Component:
    index: int
    voxel_count: int
    centroid_zyx: tuple[float, float, float]
    centroid_mm: tuple[float, float, float]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    edge_faces: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrefilteredComponent:
    component: Component
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScanFingerprint:
    slice_uid_hash: str
    volume_shape: tuple[int, int, int]
    mask_threshold: float
    marker_threshold: float
    min_marker_voxels: int
    fov_prefilter_enabled: bool
    fov_prefilter_margin: int


@dataclass(frozen=True, slots=True)
class SeedAssignment:
    component_index: int
    voxel_count: int
    centroid_zyx: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Seeds:
    schema_version: int
    fingerprint: ScanFingerprint
    assignments: dict[str, SeedAssignment]
