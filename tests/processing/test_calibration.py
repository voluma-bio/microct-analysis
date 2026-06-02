import numpy as np
import pytest

from microct_analysis.processing import profiles
from microct_analysis.processing.calibration import (
    analyze_histogram,
    derive_segmentation_thresholds,
    derive_thresholds,
)
from microct_analysis.processing.dicom import LoadError
from microct_analysis.processing.types import Thresholds


def test_analyze_histogram_detects_synthetic_air_soft_and_bone_peaks():
    rng = np.random.default_rng(42)
    volume = np.concatenate([
        rng.normal(-900, 8, 4_000),
        rng.normal(50, 8, 4_000),
        rng.normal(320, 8, 4_000),
    ])

    result = analyze_histogram(volume)

    assert result["air_peak"] is not None
    assert result["soft_tissue_peak"] is not None
    assert result["bone_peak"] is not None
    assert abs(result["air_peak"] - -900) < 25
    assert abs(result["soft_tissue_peak"] - 50) < 25
    assert abs(result["bone_peak"] - 320) < 25


def test_derive_thresholds_returns_scanco_defaults():
    thresholds = derive_thresholds(np.array([0, 1, 2]), scanner="scanco")

    assert thresholds == Thresholds()


def test_derive_thresholds_unknown_scanner_uses_histogram_peaks():
    rng = np.random.default_rng(7)
    volume = np.concatenate([
        rng.normal(-800, 5, 2_000),
        rng.normal(40, 5, 2_000),
        rng.normal(300, 5, 2_000),
    ])

    thresholds = derive_thresholds(volume, scanner="unknown")

    assert thresholds.bone_soft_tissue > 100
    assert thresholds.subchondral_cortical >= thresholds.bone_soft_tissue
    assert thresholds.surface_3d >= thresholds.subchondral_cortical


def _bimodal_segmentation_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    soft_mean: float = 2128.0,
    bone_mean: float = 3497.0,
    bone_fraction: float = 0.08,
) -> np.ndarray:
    rng = np.random.default_rng(1)
    volume = rng.normal(soft_mean, 80, size=shape).astype(np.float32)
    n_bone = int(bone_fraction * volume.size)
    indices = rng.choice(volume.size, size=n_bone, replace=False)
    flat = volume.ravel()
    flat[indices] = rng.normal(bone_mean, 60, size=n_bone)
    return flat.reshape(shape)


def test_derive_segmentation_thresholds_unknown_profile_uses_histogram_otsu() -> None:
    thresholds, analysis, flags = derive_segmentation_thresholds(_bimodal_segmentation_volume(), profiles.UNKNOWN)

    assert thresholds.method == "histogram-otsu"
    assert thresholds.mask > 2100
    assert thresholds.marker > thresholds.mask
    assert analysis.is_bimodal
    assert "calibration-unverified" in flags


def test_derive_segmentation_thresholds_scanco_profile_can_be_verified() -> None:
    thresholds, analysis, flags = derive_segmentation_thresholds(_bimodal_segmentation_volume(), profiles.SCANCO)

    assert analysis.is_bimodal
    assert thresholds.method in {"scanner-profile+histogram-verified", "histogram-otsu"}
    if thresholds.method == "histogram-otsu":
        assert "threshold-profile-disagreement" in flags


def test_derive_segmentation_thresholds_unimodal_escalates() -> None:
    rng = np.random.default_rng(2)
    volume = rng.normal(2128.0, 300, size=(64, 64, 64)).astype(np.float32)

    with pytest.raises(LoadError) as excinfo:
        derive_segmentation_thresholds(volume, profiles.UNKNOWN)

    assert excinfo.value.flag == "histogram-not-bimodal"


def test_detect_scanner_profile_from_manufacturer() -> None:
    assert profiles.detect("SCANCO Medical", "uCT 50").key == "scanco"
    assert profiles.detect("Acme", "Prototype").key == "unknown"
