"""Thresholding helpers for micro-CT volumes."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import SimpleITK as sitk


class _MaskMarkerThresholds(Protocol):
    mask: float
    marker: float


def binary_mask(volume: np.ndarray, lower_bound: int | float, upper_bound: int | float | None = None) -> np.ndarray:
    """Create a boolean mask for values within the requested intensity range."""

    source = np.asarray(volume)
    mask = source >= lower_bound
    if upper_bound is not None:
        mask &= source <= upper_bound
    return mask.astype(bool, copy=False)


def apply(volume: np.ndarray, thresholds: _MaskMarkerThresholds) -> tuple[np.ndarray, np.ndarray]:
    """Return liberal mask and morphologically opened strict markers as uint8 volumes.

    ``thresholds`` is expected to expose ``mask`` and ``marker`` fields, matching
    ``SegmentationThresholds`` produced by calibration. Marker opening uses a
    radius-1 3D binary morphological opening to remove isolated high-intensity
    noise seeds before component labeling/watershed.
    """

    source = np.asarray(volume)
    mask = (source >= thresholds.mask).astype(np.uint8)
    raw_markers = (source >= thresholds.marker).astype(np.uint8)
    markers = morphological_open(raw_markers, radius=1)
    return mask, markers


def morphological_open(binary: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Apply 3D binary morphological opening to a binary volume."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    image = sitk.GetImageFromArray(np.asarray(binary).astype(np.uint8, copy=False))
    opened = sitk.BinaryMorphologicalOpening(image, [radius, radius, radius])
    return sitk.GetArrayFromImage(opened).astype(np.uint8, copy=False)


# Backwards-compatible private alias for the mouse-ct primitive name.
_open = morphological_open
