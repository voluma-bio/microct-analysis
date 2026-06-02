"""Volume resampling helpers for micro-CT processing."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from microct_analysis.processing.dicom import LoadError


def is_isotropic(spacing: tuple[float, float, float], tol: float = 0.01) -> bool:
    """Return whether spacing is isotropic within relative tolerance."""

    _validate_spacing(spacing)
    mean_spacing = sum(spacing) / 3.0
    return all(abs(axis_spacing - mean_spacing) / mean_spacing <= tol for axis_spacing in spacing)


def anisotropy_factor(spacing: tuple[float, float, float]) -> float:
    """Return max/min voxel spacing ratio."""

    _validate_spacing(spacing)
    return max(spacing) / min(spacing)


def to_isotropic(
    volume: np.ndarray,
    spacing: tuple[float, ...],
    target_spacing: float | None = None,
    order: int = 1,
    anisotropy_limit: float = 3.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample a 3D volume to isotropic voxel spacing.

    Args:
        volume: 3D array in ``(z, y, x)`` order.
        spacing: Source voxel spacing in ``(z, y, x)`` order, in mm.
        target_spacing: Target isotropic spacing. If omitted, the smallest
            source spacing is used so the highest input resolution is preserved.
        order: Interpolation order compatibility knob. ``0`` maps to nearest
            neighbor, ``1`` maps to linear interpolation, and ``2``-``5`` map
            to SimpleITK B-spline interpolators of the same order.
        anisotropy_limit: Maximum allowed max/min spacing ratio. Larger ratios
            raise ``LoadError('anisotropy-too-large', ...)``.

    Returns:
        ``(resampled_volume, new_spacing)`` where ``new_spacing`` repeats the
        isotropic target spacing for each input dimension. Already isotropic
        input is returned by identity when ``target_spacing`` is omitted.
    """

    spacing_zyx = _normalize_spacing(spacing)
    source = np.asarray(volume)
    if source.ndim != 3:
        raise ValueError("volume must be a 3D array")
    if anisotropy_limit <= 0:
        raise ValueError("anisotropy_limit must be positive")

    if target_spacing is None and is_isotropic(spacing_zyx):
        return source, spacing_zyx

    factor = anisotropy_factor(spacing_zyx)
    if factor > anisotropy_limit:
        raise LoadError(
            "anisotropy-too-large",
            f"anisotropy factor {factor:.2f}x exceeds limit {anisotropy_limit:.2f}x",
        )

    resolved_target = min(spacing_zyx) if target_spacing is None else float(target_spacing)
    if resolved_target <= 0:
        raise ValueError("target_spacing must be positive")

    image = sitk.GetImageFromArray(source)
    # SimpleITK uses (x, y, z) spacing/size while arrays are (z, y, x).
    image.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))

    original_size = np.asarray(image.GetSize(), dtype=np.int64)
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_size = np.maximum(1, np.round(original_size * original_spacing / resolved_target).astype(int)).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((resolved_target, resolved_target, resolved_target))
    resampler.SetSize([int(size) for size in new_size])
    resampler.SetInterpolator(_interpolator(order))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampled = resampler.Execute(image)
    resampled_array = sitk.GetArrayFromImage(resampled)
    if order != 0 and np.issubdtype(source.dtype, np.floating):
        resampled_array = resampled_array.astype(source.dtype, copy=False)
    return np.asarray(resampled_array), (resolved_target, resolved_target, resolved_target)


def _normalize_spacing(spacing: tuple[float, ...]) -> tuple[float, float, float]:
    if len(spacing) != 3:
        raise ValueError("spacing must contain three values in (z, y, x) order")
    spacing_zyx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    _validate_spacing(spacing_zyx)
    return spacing_zyx


def _validate_spacing(spacing: tuple[float, float, float]) -> None:
    if any(axis_spacing <= 0 for axis_spacing in spacing):
        raise ValueError("spacing values must be positive")


def _interpolator(order: int) -> int:
    interpolators = {
        0: sitk.sitkNearestNeighbor,
        1: sitk.sitkLinear,
        2: sitk.sitkBSpline2,
        3: sitk.sitkBSpline3,
        4: sitk.sitkBSpline4,
        5: sitk.sitkBSpline5,
    }
    try:
        return int(interpolators[order])
    except KeyError as exc:
        raise ValueError("order must be an integer from 0 through 5") from exc
