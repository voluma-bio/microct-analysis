"""Preprocessing filters for micro-CT volumes."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def median_filter(
    volume: np.ndarray,
    *,
    iterations: int = 3,
    size: int = 3,
    radius: int | None = None,
) -> np.ndarray:
    """Apply an iterative XY-plane median filter slice-by-slice.

    The Amira SOP uses a 3-iteration median filter in the XY plane.  The
    first axis is treated as slice/depth, so no median window spans adjacent
    slices.

    ``size`` preserves the existing scipy-style API exactly when ``radius`` is
    omitted, including even window sizes and dtype preservation. ``radius`` is
    accepted for compatibility with the SimpleITK/mouse-ct primitive where
    ``radius=1`` is equivalent to a 3x3 in-plane window.
    """

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if size < 1:
        raise ValueError("size must be positive")
    if radius is not None and radius < 0:
        raise ValueError("radius must be non-negative")

    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must be a 3D array")
    if iterations == 0:
        return source.copy()
    if radius is None:
        return _median_filter_by_size(source, iterations=iterations, size=size)
    return _median_filter_by_radius(source, iterations=iterations, radius=int(radius))


def _median_filter_by_size(volume: np.ndarray, *, iterations: int, size: int) -> np.ndarray:
    filtered = volume.copy()
    for _ in range(iterations):
        filtered = ndimage.median_filter(filtered, size=(1, size, size))
    return filtered


def _median_filter_by_radius(volume: np.ndarray, *, iterations: int, radius: int) -> np.ndarray:
    filtered = volume.astype(np.float32, copy=True)
    for _ in range(iterations):
        slices: list[np.ndarray] = []
        for z in range(filtered.shape[0]):
            image = sitk.GetImageFromArray(filtered[z])
            median = sitk.Median(image, [radius, radius])
            slices.append(sitk.GetArrayFromImage(median))
        filtered = np.stack(slices, axis=0).astype(np.float32, copy=False)
    return filtered
