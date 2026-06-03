"""Histogram-based calibration and threshold defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from microct_analysis.processing.dicom import LoadError
from microct_analysis.processing.profiles import ScannerProfile, get
from microct_analysis.processing.types import SegmentationThresholds, Thresholds

HIST_BINS = 512
HIST_RANGE = (-500.0, 20000.0)
DEFAULT_BIMODALITY_RATIO = 0.55
DEFAULT_MARKER_PERCENTILE = 65.0
VALLEY_PEAK_RATIO = 1.3


@dataclass(frozen=True)
class HistogramAnalysis:
    """Histogram diagnostics used by segmentation threshold derivation."""

    counts: np.ndarray
    bin_edges: np.ndarray
    otsu_threshold: float
    separability: float
    is_bimodal: bool


def analyze_histogram(volume: np.ndarray) -> dict[str, Any]:
    """Detect air, soft-tissue, and bone peaks in an intensity histogram.

    This legacy helper is kept for intake reports and current workflow tests.
    Segmentation uses :func:`analyze_segmentation_histogram` instead.
    """

    values = _finite_values(volume)
    if values.size == 0:
        return _empty_peak_result()

    counts, centers = _legacy_histogram(values)
    if counts.max() == 0:
        return _empty_peak_result()

    normalized = counts / counts.max()
    peak_indices, properties = find_peaks(normalized, prominence=0.01)
    peaks = sorted(float(centers[index]) for index in peak_indices)

    return {
        "air_peak": peaks[0] if len(peaks) > 0 else None,
        "soft_tissue_peak": peaks[1] if len(peaks) > 1 else None,
        "bone_peak": peaks[-1] if len(peaks) > 0 else None,
        "peaks": peaks,
        "histogram": {"counts": counts.tolist(), "bin_centers": centers.tolist()},
        "prominences": properties.get("prominences", np.array([], dtype=float)).tolist(),
    }


def derive_thresholds(volume: np.ndarray, scanner: str = "scanco") -> Thresholds:
    """Derive legacy SCANCO-equivalent processing thresholds."""

    if scanner.lower() == "scanco":
        return Thresholds()
    return _legacy_thresholds_from_peaks(analyze_histogram(volume))


def analyze_segmentation_histogram(
    volume: np.ndarray,
    *,
    bimodality_ratio: float = DEFAULT_BIMODALITY_RATIO,
    subsample_stride: int = 4,
) -> HistogramAnalysis:
    """Analyze a volume histogram for segmentation threshold derivation.

    The algorithm matches the working mouse-ct pipeline: subsample, build a
    fixed-range 512-bin histogram, choose the Otsu threshold, and gate on both
    Otsu separability and valley depth.
    """

    counts, bin_edges = _segmentation_histogram(volume, subsample_stride=subsample_stride)
    total = int(counts.sum())
    if total == 0:
        return HistogramAnalysis(counts, bin_edges, float("nan"), 0.0, False)

    centers = _bin_centers(bin_edges)
    best_index, separability = _otsu_index_and_separability(counts, centers)
    otsu = float(centers[best_index])
    is_bimodal = separability >= bimodality_ratio and _has_bimodal_valley(counts, best_index)

    return HistogramAnalysis(
        counts=counts,
        bin_edges=bin_edges,
        otsu_threshold=otsu,
        separability=separability,
        is_bimodal=is_bimodal,
    )


def derive_segmentation_thresholds(
    volume: np.ndarray,
    profile: ScannerProfile | str = "unknown",
    *,
    bimodality_ratio: float = DEFAULT_BIMODALITY_RATIO,
    marker_percentile: float = DEFAULT_MARKER_PERCENTILE,
) -> tuple[SegmentationThresholds, HistogramAnalysis, list[str]]:
    """Derive segmentation mask/marker thresholds from histogram and scanner profile."""

    scanner_profile = get(profile) if isinstance(profile, str) else profile
    analysis = analyze_segmentation_histogram(volume, bimodality_ratio=bimodality_ratio)
    _require_bimodal(analysis, bimodality_ratio)

    histogram_thresholds = _histogram_segmentation_thresholds(volume, analysis, marker_percentile)
    flags: list[str] = []
    if scanner_profile.key == "unknown":
        flags.append("calibration-unverified")
    return histogram_thresholds, analysis, flags


def _finite_values(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _empty_peak_result() -> dict[str, Any]:
    return {"air_peak": None, "soft_tissue_peak": None, "bone_peak": None, "peaks": []}


def _legacy_histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=256)
    return counts, _bin_centers(edges)


def _legacy_thresholds_from_peaks(observations: dict[str, Any]) -> Thresholds:
    soft = observations.get("soft_tissue_peak")
    bone = observations.get("bone_peak")
    if soft is None or bone is None or bone <= soft:
        return Thresholds()

    bone_soft_tissue = int(round((float(soft) + float(bone)) / 2.0))
    subchondral_cortical = int(round(float(bone) * 0.9 + bone_soft_tissue * 0.1))
    surface_3d = int(round(float(bone) * 1.05))
    return Thresholds(
        bone_soft_tissue=bone_soft_tissue,
        subchondral_cortical=max(subchondral_cortical, bone_soft_tissue),
        surface_3d=max(surface_3d, subchondral_cortical),
        marrow_bone=max(subchondral_cortical, bone_soft_tissue),
    )


def _segmentation_histogram(volume: np.ndarray, *, subsample_stride: int) -> tuple[np.ndarray, np.ndarray]:
    if subsample_stride < 1:
        raise ValueError("subsample_stride must be >= 1")
    samples = np.asarray(volume, dtype=np.float32)[::subsample_stride, ::subsample_stride, ::subsample_stride]
    return np.histogram(samples.ravel(), bins=HIST_BINS, range=HIST_RANGE)


def _bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def _otsu_index_and_separability(counts: np.ndarray, centers: np.ndarray) -> tuple[int, float]:
    probability = counts.astype(np.float64) / int(counts.sum())
    var_total = _total_variance(probability, centers)
    cum_probability = np.cumsum(probability)
    cum_mean = np.cumsum(centers * probability)

    best_separability = -1.0
    best_index = 0
    for index in range(1, len(counts) - 1):
        separability = _between_class_variance(index, cum_probability, cum_mean)
        if separability > best_separability:
            best_separability = float(separability)
            best_index = index

    normalized_separability = float(best_separability / var_total) if var_total > 0 else 0.0
    return best_index, normalized_separability


def _total_variance(probability: np.ndarray, centers: np.ndarray) -> float:
    mean_total = float(np.sum(centers * probability))
    return float(np.sum(((centers - mean_total) ** 2) * probability))


def _between_class_variance(index: int, cum_probability: np.ndarray, cum_mean: np.ndarray) -> float:
    weight_left = cum_probability[index]
    weight_right = 1.0 - weight_left
    if weight_left <= 0.0 or weight_right <= 0.0:
        return -1.0
    mean_left = cum_mean[index] / weight_left
    mean_right = (cum_mean[-1] - cum_mean[index]) / weight_right
    return float(weight_left * weight_right * (mean_left - mean_right) ** 2)


def _has_bimodal_valley(counts: np.ndarray, valley_index: int) -> bool:
    valley_count = max(int(counts[valley_index]), 1)
    left_peak = int(counts[:valley_index].max()) if valley_index > 0 else 0
    right_peak = int(counts[valley_index + 1 :].max()) if valley_index + 1 < len(counts) else 0
    return left_peak >= VALLEY_PEAK_RATIO * valley_count and right_peak >= VALLEY_PEAK_RATIO * valley_count


def _require_bimodal(analysis: HistogramAnalysis, bimodality_ratio: float) -> None:
    if analysis.is_bimodal:
        return
    raise LoadError(
        "histogram-not-bimodal",
        f"histogram separability {analysis.separability:.3f} below bimodality ratio {bimodality_ratio}",
    )


def _histogram_segmentation_thresholds(
    volume: np.ndarray,
    analysis: HistogramAnalysis,
    marker_percentile: float,
) -> SegmentationThresholds:
    mask_threshold = analysis.otsu_threshold
    volume_array = np.asarray(volume)
    above_mask = volume_array[volume_array >= mask_threshold]
    if above_mask.size == 0:
        raise LoadError("histogram-not-bimodal", "no voxels above Otsu threshold (degenerate histogram)")
    marker_threshold = float(np.percentile(above_mask, marker_percentile))
    return SegmentationThresholds(mask=mask_threshold, marker=marker_threshold, method="histogram-otsu")
