"""Seeded segmentation and marker-labeling primitives for micro-CT volumes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import cast

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from microct_analysis.processing.dicom import LoadError
from microct_analysis.processing.types import (
    BONE_LABELS,
    REQUIRED_SEED_BONES,
    Component,
    PrefilteredComponent,
    ScanFingerprint,
    SeedAssignment,
    Seeds,
)

DEFAULT_MIN_MARKER_VOXELS = 500
SIMILAR_VOLUME_FRACTION = 0.10
PATELLA_SESAMOID_RATIO = 0.50
BRIDGE_MARKER_ADJACENCY_MM = 0.025

FOV_EDGE_VOXEL_FRACTION = 0.50
FOV_SHAFT_Z_FRACTION = 0.30
FOV_THIN_SHEET_RATIO = 3.0
FOV_SPARSE_RATIO = 5.0


@dataclass(frozen=True)
class FovPrefilterConfig:
    enabled: bool = True
    margin_voxels: int = 3


@dataclass(frozen=True)
class MarkerLabeling:
    labeled_markers: np.ndarray
    bone_assignments: dict[str, int]


def seed_from_region(
    filtered: np.ndarray,
    mask: np.ndarray,
    label_assignments: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """Create marker seeds by flood filling masked components from centroids.

    Each assignment receives a stable 1-based marker ID in insertion order.  A
    seed point must lie inside ``mask``; the marker is expanded to the connected
    masked component containing that point, approximating Amira's Magic Wand / All
    Slices behavior.
    """

    volume_shape = np.asarray(filtered).shape
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != volume_shape:
        raise ValueError("mask shape must match filtered volume shape")

    markers = np.zeros(volume_shape, dtype=np.int32)
    components, _ = cast(tuple[np.ndarray, int], ndimage.label(mask_array))
    for label_id, seed in enumerate(label_assignments.values(), start=1):
        if len(seed) != len(volume_shape):
            raise ValueError("seed dimensionality must match volume")
        seed_index = tuple(int(coord) for coord in seed[: len(volume_shape)])
        if any(coord < 0 or coord >= limit for coord, limit in zip(seed_index, volume_shape, strict=True)):
            raise ValueError(f"seed {seed_index} is outside volume bounds")
        component_id = int(components[seed_index])
        if component_id == 0:
            raise ValueError(f"seed {seed_index} is outside mask")
        markers[components == component_id] = label_id
    return markers


def watershed_segment(filtered: np.ndarray, seeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run marker-based watershed inside ``mask`` on the intensity volume."""

    filtered_array = np.asarray(filtered)
    seeds_array = np.asarray(seeds, dtype=np.int32)
    mask_array = np.asarray(mask, dtype=bool)
    if filtered_array.shape != seeds_array.shape or filtered_array.shape != mask_array.shape:
        raise ValueError("filtered, seeds, and mask must have matching shapes")

    inverted = -filtered_array.astype(np.float32, copy=False)
    intensity_image = sitk.GetImageFromArray(inverted)
    seed_image = sitk.GetImageFromArray(seeds_array.astype(np.int16, copy=False))
    grown = sitk.MorphologicalWatershedFromMarkers(
        intensity_image,
        seed_image,
        False,
        True,
    )
    grown_array = sitk.GetArrayFromImage(grown).astype(np.int32, copy=False)
    grown_array[~mask_array] = 0
    return _keep_marker_connected_components(grown_array, seeds_array)


def extract_label(grown: np.ndarray, label_id: int) -> np.ndarray:
    """Extract one integer label as a boolean mask."""

    return np.asarray(grown) == label_id


def connected_components(binary: np.ndarray) -> np.ndarray:
    """Fully-connected connected-component labeling with background=0."""

    image = sitk.GetImageFromArray(np.asarray(binary).astype(np.uint8, copy=False))
    cc = sitk.ConnectedComponent(image, True)
    return sitk.GetArrayFromImage(cc).astype(np.int32, copy=False)


_connected_components = connected_components


def extract_components(
    marker_binary: np.ndarray,
    spacing: tuple[float, float, float],
    min_voxels: int,
) -> tuple[np.ndarray, list[Component], int]:
    """Run component labeling and build stats for marker components."""

    if min_voxels < 1:
        raise ValueError("min_voxels must be positive")
    marker_array = np.asarray(marker_binary)
    if marker_array.ndim != 3:
        raise ValueError("marker_binary must be a 3D array")

    cc = connected_components(marker_array)
    max_label = int(cc.max())
    if max_label == 0:
        return cc, [], 0

    sz, sy, sx = spacing
    counts = np.bincount(cc.ravel(), minlength=max_label + 1)

    components: list[Component] = []
    for label in range(1, max_label + 1):
        voxel_count = int(counts[label])
        if voxel_count < min_voxels:
            continue
        coords = np.argwhere(cc == label)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        centroid_z = float(coords[:, 0].mean())
        centroid_y = float(coords[:, 1].mean())
        centroid_x = float(coords[:, 2].mean())
        bbox = (
            (int(z_min), int(z_max) + 1),
            (int(y_min), int(y_max) + 1),
            (int(x_min), int(x_max) + 1),
        )
        components.append(
            Component(
                index=label,
                voxel_count=voxel_count,
                centroid_zyx=(centroid_z, centroid_y, centroid_x),
                centroid_mm=(centroid_z * sz, centroid_y * sy, centroid_x * sx),
                bbox_zyx=bbox,
                edge_faces=_edge_faces_for(bbox, marker_array.shape),
            )
        )
    components.sort(key=lambda component: component.voxel_count, reverse=True)
    return cc, components, max_label


def fov_prefilter(
    components: list[Component],
    cc_array: np.ndarray,
    shape: tuple[int, int, int],
    cfg: FovPrefilterConfig,
) -> tuple[list[Component], list[PrefilteredComponent]]:
    """Drop edge-hugging, non-shaft, non-compact marker components."""

    if not cfg.enabled:
        return list(components), []
    retained: list[Component] = []
    prefiltered: list[PrefilteredComponent] = []
    for component in components:
        drop, reasons = _classify_prefilter(component, cc_array, shape, cfg)
        if drop:
            prefiltered.append(PrefilteredComponent(component=component, reasons=reasons))
        else:
            retained.append(component)
    return retained, prefiltered


def heuristic_assign(components: list[Component], z_extent: int) -> dict[str, int]:
    """Assign bone identity by size and position rules; raise on ambiguity."""

    if len(components) < 3:
        raise LoadError("ambiguous-bone-identity", f"only {len(components)} large marker components (expected ≥3)")

    superior = [component for component in components if _z_half(component.centroid_zyx[0], z_extent) == "superior"]
    inferior = [component for component in components if _z_half(component.centroid_zyx[0], z_extent) == "inferior"]
    if not superior or not inferior:
        raise LoadError("ambiguous-bone-identity", "components do not span both superior and inferior halves")

    femur = superior[0]
    tibia = inferior[0]

    if _has_similar_in_half(superior, femur) or _has_similar_in_half(inferior, tibia):
        raise LoadError("ambiguous-bone-identity", "two similar-volume components in the same anatomical half")

    z_third = z_extent / 3.0
    for component in components:
        (z_min, z_max), _, _ = component.bbox_zyx
        span = z_max - z_min
        if span > z_extent / 2 and z_min < z_third and z_max > 2 * z_third:
            raise LoadError(
                "ambiguous-bone-identity",
                f"component {component.index} spans {span} slices across both z-thirds",
            )

    remaining = [component for component in components if component.index not in (femur.index, tibia.index)]
    midline_y = 0.5 * (femur.centroid_zyx[1] + tibia.centroid_zyx[1])
    midline_x = 0.5 * (femur.centroid_zyx[2] + tibia.centroid_zyx[2])

    anterior = sorted(remaining, key=lambda component: component.centroid_zyx[1])
    patella = anterior[0] if anterior and anterior[0].centroid_zyx[1] < midline_y else None

    lateral = sorted(remaining, key=lambda component: -abs(component.centroid_zyx[2] - midline_x))
    fibula_candidates = [
        component for component in lateral if component is not patella and abs(component.centroid_zyx[2] - midline_x) > 0
    ]
    fibula = fibula_candidates[0] if fibula_candidates else None

    assigned_ids = {femur.index, tibia.index}
    mapping: dict[str, int] = {"femur": femur.index, "tibia": tibia.index}
    if patella is not None:
        mapping["patella"] = patella.index
        assigned_ids.add(patella.index)
    if fibula is not None and fibula.index not in assigned_ids:
        mapping["fibula"] = fibula.index
        assigned_ids.add(fibula.index)

    others = [component for component in components if component.index not in assigned_ids]
    if patella is not None:
        for component in others:
            if component.voxel_count > PATELLA_SESAMOID_RATIO * patella.voxel_count:
                raise LoadError(
                    "ambiguous-bone-identity",
                    f"extra component ({component.voxel_count} voxels) > "
                    f"{PATELLA_SESAMOID_RATIO:.0%} of patella ({patella.voxel_count})",
                )

    for offset, component in enumerate(others, start=5):
        mapping[f"other-{offset}"] = component.index

    return mapping


def propose_auto_seeds(components: list[Component], z_extent: int) -> dict[str, SeedAssignment]:
    """Deterministically propose femur/tibia seed anchors from components."""

    ordered = sorted(components, key=lambda component: (-component.voxel_count, component.index))
    half = z_extent / 2
    superior = [component for component in ordered if component.centroid_zyx[0] < half]
    inferior = [component for component in ordered if component.centroid_zyx[0] >= half]

    assignments: dict[str, SeedAssignment] = {}
    if superior:
        component = superior[0]
        assignments["femur"] = SeedAssignment(
            component_index=component.index,
            voxel_count=component.voxel_count,
            centroid_zyx=component.centroid_zyx,
        )
    if inferior:
        component = inferior[0]
        assignments["tibia"] = SeedAssignment(
            component_index=component.index,
            voxel_count=component.voxel_count,
            centroid_zyx=component.centroid_zyx,
        )
    return assignments


def assign_from_seeds(
    components: list[Component],
    seeds: Seeds,
    current_fingerprint: ScanFingerprint,
) -> dict[str, int]:
    """Validate seeds against the current run and return a bone→component-index mapping."""

    if seeds.schema_version != 1:
        raise LoadError("invalid-seeds", f"unsupported seeds schema_version {seeds.schema_version!r}")

    fp = seeds.fingerprint
    if fp.slice_uid_hash != current_fingerprint.slice_uid_hash:
        raise LoadError(
            "invalid-seeds",
            f"slice_uid_hash mismatch (seeds={fp.slice_uid_hash!r} current={current_fingerprint.slice_uid_hash!r})",
        )
    if tuple(fp.volume_shape) != tuple(current_fingerprint.volume_shape):
        raise LoadError(
            "invalid-seeds",
            f"volume_shape mismatch (seeds={fp.volume_shape} current={current_fingerprint.volume_shape})",
        )
    if not _float_close(fp.mask_threshold, current_fingerprint.mask_threshold, 0.01):
        raise LoadError(
            "invalid-seeds",
            f"mask_threshold drift (seeds={fp.mask_threshold} current={current_fingerprint.mask_threshold})",
        )
    if not _float_close(fp.marker_threshold, current_fingerprint.marker_threshold, 0.01):
        raise LoadError(
            "invalid-seeds",
            f"marker_threshold drift (seeds={fp.marker_threshold} current={current_fingerprint.marker_threshold})",
        )
    if fp.min_marker_voxels != current_fingerprint.min_marker_voxels:
        raise LoadError(
            "invalid-seeds",
            f"min_marker_voxels mismatch (seeds={fp.min_marker_voxels} current={current_fingerprint.min_marker_voxels})",
        )
    if fp.fov_prefilter_enabled != current_fingerprint.fov_prefilter_enabled:
        raise LoadError("invalid-seeds", "fov_prefilter enabled-flag mismatch")
    if fp.fov_prefilter_margin != current_fingerprint.fov_prefilter_margin:
        raise LoadError(
            "invalid-seeds",
            f"fov_prefilter_margin mismatch (seeds={fp.fov_prefilter_margin} "
            f"current={current_fingerprint.fov_prefilter_margin})",
        )

    missing = REQUIRED_SEED_BONES - set(seeds.assignments)
    if missing:
        raise LoadError("invalid-seeds", f"required bones missing from seeds: {sorted(missing)}")

    seen_indices: dict[int, str] = {}
    components_by_index = {component.index: component for component in components}
    mapping: dict[str, int] = {}
    for bone, assignment in seeds.assignments.items():
        if not _valid_bone_name(bone):
            raise LoadError("invalid-seeds", f"unknown bone name {bone!r}")
        if assignment.component_index in seen_indices:
            other = seen_indices[assignment.component_index]
            raise LoadError(
                "invalid-seeds",
                f"component_index {assignment.component_index} used by both {other!r} and {bone!r}",
            )
        component = components_by_index.get(assignment.component_index)
        if component is None:
            raise LoadError(
                "invalid-seeds",
                f"component_index {assignment.component_index} for {bone!r} not in current retained components",
            )
        if not _float_close(float(assignment.voxel_count), float(component.voxel_count), 0.02):
            raise LoadError(
                "invalid-seeds",
                f"{bone!r} voxel_count drift (seeds={assignment.voxel_count}, current={component.voxel_count})",
            )
        for axis, (seed_coord, current_coord) in enumerate(zip(assignment.centroid_zyx, component.centroid_zyx, strict=True)):
            if abs(seed_coord - current_coord) > 1.0:
                raise LoadError(
                    "invalid-seeds",
                    f"{bone!r} centroid drift on axis {axis} "
                    f"(seeds={seed_coord:.2f}, current={current_coord:.2f})",
                )
        seen_indices[assignment.component_index] = bone
        mapping[bone] = assignment.component_index

    return mapping


def paint_and_verify(
    cc_array: np.ndarray,
    assignments: dict[str, int],
    spacing: tuple[float, float, float],
) -> MarkerLabeling:
    """Paint canonical labels and run pre-watershed bridge detection."""

    _check_mask_bridge(cc_array, assignments, spacing)
    return MarkerLabeling(labeled_markers=_paint_canonical(cc_array, assignments), bone_assignments=assignments)


def label_bones(
    marker_binary: np.ndarray,
    spacing: tuple[float, float, float],
    min_marker_voxels: int = DEFAULT_MIN_MARKER_VOXELS,
) -> MarkerLabeling:
    """Convenience helper: extract components, heuristic-assign, paint, and verify."""

    cc_array, components, _ = extract_components(marker_binary, spacing, min_marker_voxels)
    assignments = heuristic_assign(components, np.asarray(marker_binary).shape[0])
    return paint_and_verify(cc_array, assignments, spacing)


def _keep_marker_connected_components(labels: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Remove watershed label components not connected to their marker seeds."""

    cleaned = np.zeros_like(labels)
    for raw_label in np.unique(labels):
        label = int(raw_label)
        if label == 0:
            continue
        marker_for_label = markers == label
        if not np.any(marker_for_label):
            continue
        label_cc = connected_components(labels == label)
        hit = np.unique(label_cc[marker_for_label])
        hit_ids = [int(component_id) for component_id in hit if component_id != 0]
        if hit_ids:
            cleaned[np.isin(label_cc, hit_ids)] = label
    return cleaned


def _edge_faces_for(
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    shape: tuple[int, int, int],
) -> tuple[str, ...]:
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    z, y, x = shape
    faces: list[str] = []
    if z_min <= 0:
        faces.append("z_min")
    if z_max >= z:
        faces.append("z_max")
    if y_min <= 0:
        faces.append("y_min")
    if y_max >= y:
        faces.append("y_max")
    if x_min <= 0:
        faces.append("x_min")
    if x_max >= x:
        faces.append("x_max")
    return tuple(faces)


def _classify_prefilter(
    component: Component,
    cc_array: np.ndarray,
    shape: tuple[int, int, int],
    cfg: FovPrefilterConfig,
) -> tuple[bool, tuple[str, ...]]:
    _, y, x = shape
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = component.bbox_zyx

    touches_xy = any(face in component.edge_faces for face in ("x_min", "x_max", "y_min", "y_max"))
    if not touches_xy:
        return False, ()

    margin = max(1, cfg.margin_voxels)
    sub_mask = cc_array[z_min:z_max, y_min:y_max, x_min:x_max] == component.index
    if sub_mask.size == 0 or not sub_mask.any():
        return False, ()
    coords = np.argwhere(sub_mask)
    global_y = coords[:, 1] + y_min
    global_x = coords[:, 2] + x_min
    near_edge = (global_y < margin) | (global_y >= y - margin) | (global_x < margin) | (global_x >= x - margin)
    edge_fraction = float(near_edge.mean())
    if edge_fraction <= FOV_EDGE_VOXEL_FRACTION:
        return False, ()

    z_extent = z_max - z_min
    if z_extent >= FOV_SHAFT_Z_FRACTION * shape[0]:
        return False, ()

    y_extent = max(1, y_max - y_min)
    x_extent = max(1, x_max - x_min)
    xy_aspect = max(y_extent, x_extent) / min(y_extent, x_extent)
    bbox_volume = max(1, z_extent) * y_extent * x_extent
    sparse_ratio = bbox_volume / max(1, component.voxel_count)
    if xy_aspect <= FOV_THIN_SHEET_RATIO and sparse_ratio <= FOV_SPARSE_RATIO:
        return False, ()

    reasons = [f"edge-fraction={edge_fraction:.2f}", f"z-extent={z_extent}/{shape[0]}"]
    if xy_aspect > FOV_THIN_SHEET_RATIO:
        reasons.append(f"xy-aspect={xy_aspect:.2f}")
    if sparse_ratio > FOV_SPARSE_RATIO:
        reasons.append(f"sparse-ratio={sparse_ratio:.2f}")
    return True, tuple(reasons)


def _z_half(centroid_z: float, z_extent: int) -> str:
    return "superior" if centroid_z < z_extent / 2 else "inferior"


def _has_similar_in_half(half: list[Component], primary: Component) -> bool:
    for component in half:
        if component is primary:
            continue
        if abs(component.voxel_count - primary.voxel_count) / primary.voxel_count <= SIMILAR_VOLUME_FRACTION:
            return True
    return False


def _valid_bone_name(name: str) -> bool:
    if name in {"femur", "tibia", "fibula", "patella"}:
        return True
    if name.startswith("other-"):
        try:
            return int(name.split("-", 1)[1]) >= 5
        except ValueError:
            return False
    return False


def _float_close(a: float, b: float, rel: float) -> bool:
    if b == 0:
        return abs(a) <= rel
    return abs(a - b) / abs(b) <= rel


def _paint_canonical(cc_array: np.ndarray, assignments: dict[str, int]) -> np.ndarray:
    name_to_label = {name: label for label, name in BONE_LABELS.items()}
    out = np.zeros_like(cc_array, dtype=np.uint8)
    next_other = max(name_to_label.values()) + 1
    for name, cc_idx in assignments.items():
        if name in name_to_label:
            label = name_to_label[name]
        else:
            label = next_other
            next_other += 1
        out[cc_array == cc_idx] = label
    return out


def _check_mask_bridge(cc_array: np.ndarray, assignments: dict[str, int], spacing: tuple[float, float, float]) -> None:
    if "femur" not in assignments or "tibia" not in assignments:
        return
    femur_marker = (cc_array == assignments["femur"]).astype(np.uint8)
    tibia_marker = cc_array == assignments["tibia"]
    if femur_marker.sum() == 0 or tibia_marker.sum() == 0:
        return

    radius = max(1, ceil(BRIDGE_MARKER_ADJACENCY_MM / min(spacing)))
    image = sitk.GetImageFromArray(femur_marker)
    dilated = sitk.BinaryDilate(image, [radius, radius, radius])
    dilated_array = sitk.GetArrayFromImage(dilated).astype(bool, copy=False)
    if np.any(dilated_array & tibia_marker):
        raise LoadError(
            "articular-bridging-suspected",
            f"femur and tibia marker components are within {BRIDGE_MARKER_ADJACENCY_MM * 1000:.0f} μm "
            f"({radius} voxels), indicating possible cortical fusion across the joint",
        )
