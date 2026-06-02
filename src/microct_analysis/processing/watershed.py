"""Marker-based watershed and post-processing helpers."""

from __future__ import annotations

import dataclasses

import numpy as np
import SimpleITK as sitk

from microct_analysis.processing.segmentation import connected_components
from microct_analysis.processing.types import BONE_LABELS, BoneLabels, BoneStats


def run(
    filtered_volume: np.ndarray,
    labeled_markers: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> BoneLabels:
    """Grow labeled markers through the filtered intensity volume inside ``mask``."""

    filtered = np.asarray(filtered_volume)
    markers = np.asarray(labeled_markers)
    mask_array = np.asarray(mask)
    if filtered.shape != markers.shape or filtered.shape != mask_array.shape:
        raise ValueError("filtered_volume, labeled_markers, and mask must have matching shapes")

    inverted = -filtered.astype(np.float32, copy=False)
    intensity_image = sitk.GetImageFromArray(inverted)
    marker_image = sitk.GetImageFromArray(markers.astype(np.int16, copy=False))
    intensity_image.SetSpacing((spacing[2], spacing[1], spacing[0]))
    marker_image.SetSpacing((spacing[2], spacing[1], spacing[0]))

    grown = sitk.MorphologicalWatershedFromMarkers(
        intensity_image,
        marker_image,
        False,
        True,
    )
    grown_array = sitk.GetArrayFromImage(grown).astype(np.uint8, copy=False)
    grown_array *= (mask_array > 0).astype(np.uint8, copy=False)
    grown_array = _keep_marker_connected_components(grown_array, markers)
    return BoneLabels(volume=grown_array, per_bone=describe_bones(grown_array, spacing), flags=[])


def prune_to_seed_cc(
    bone_labels: BoneLabels,
    labeled_markers: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[BoneLabels, dict[str, int], list[str]]:
    """Keep only watershed components connected to each bone's original marker seeds."""

    volume = np.asarray(bone_labels.volume)
    cleaned = np.zeros_like(volume)
    pruning_stats: dict[str, int] = {}
    flags: list[str] = []

    for raw_label in np.unique(volume):
        label = int(raw_label)
        if label == 0:
            continue
        bone_mask = (volume == label).astype(np.uint8)
        before = int(bone_mask.sum())
        cc = connected_components(bone_mask)
        marker_for_label = np.asarray(labeled_markers) == label
        name = BONE_LABELS.get(label, f"label{label}")
        if not np.any(marker_for_label):
            flags.append(f"pruning-dropped-bone:{name}")
            pruning_stats[name] = before
            continue
        hit = np.unique(cc[marker_for_label])
        hit_ids = [int(component_id) for component_id in hit if component_id != 0]
        if not hit_ids:
            flags.append(f"pruning-dropped-bone:{name}")
            pruning_stats[name] = before
            continue
        keep = np.isin(cc, hit_ids)
        cleaned[keep] = label
        pruning_stats[name] = before - int(keep.sum())

    new_per_bone = describe_bones(cleaned, spacing)
    return dataclasses.replace(bone_labels, volume=cleaned, per_bone=new_per_bone), pruning_stats, flags


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


def describe_bones(labels: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, BoneStats]:
    """Describe non-background label volumes with counts, volume, centroid, and bbox."""

    sz, sy, sx = spacing
    voxel_volume = sz * sy * sx
    out: dict[str, BoneStats] = {}
    labels_array = np.asarray(labels)
    for raw_label in np.unique(labels_array):
        label = int(raw_label)
        if label == 0:
            continue
        coords = np.argwhere(labels_array == label)
        if coords.size == 0:
            continue
        voxel_count = int(coords.shape[0])
        centroid_z = float(coords[:, 0].mean()) * sz
        centroid_y = float(coords[:, 1].mean()) * sy
        centroid_x = float(coords[:, 2].mean()) * sx
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        name = BONE_LABELS.get(label, f"other-{label}")
        out[name] = BoneStats(
            label=label,
            name=name,
            voxel_count=voxel_count,
            volume_mm3=float(voxel_count * voxel_volume),
            centroid_mm=(centroid_z, centroid_y, centroid_x),
            bbox_zyx=(
                (int(z_min), int(z_max) + 1),
                (int(y_min), int(y_max) + 1),
                (int(x_min), int(x_max) + 1),
            ),
        )
    return out


_describe_bones = describe_bones
