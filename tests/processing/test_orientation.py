"""Tests for micro-CT orientation helpers.

PCA orientation has been retired in favour of visual landmarking.
The orientation.py module has been removed; these tests are preserved
as a record of the prior behaviour.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="PCA orientation retired in visual-landmarking")


import numpy as np


def _principal_axis(mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    positions = np.argwhere(mask != 0).astype(np.float64) * np.asarray(spacing)
    centered = positions - positions.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = vectors[:, np.argmax(values)]
    return np.abs(axis)


def _object_center(mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    positions = np.argwhere(mask != 0).astype(np.float64) * np.asarray(spacing)
    return (positions.min(axis=0) + positions.max(axis=0)) / 2.0


def test_center_volume_computes_translation_for_off_center_object() -> None:
    from microct_analysis.processing.orientation import center_volume

    volume = np.zeros((10, 12, 14), dtype=np.uint8)
    volume[2:5, 4:8, 6:10] = 1
    spacing = (0.5, 1.0, 2.0)

    translated, translation = center_volume(volume, spacing)

    np.testing.assert_allclose(translation, np.array([0.75, 0.0, -2.0]))
    np.testing.assert_allclose(_object_center(translated, spacing), np.array([2.25, 5.5, 13.0]), atol=0.25)


def test_pca_orient_elongated_ellipsoid_aligns_pc1_with_z() -> None:
    from microct_analysis.processing.orientation import pca_orient

    spacing = (1.0, 1.0, 1.0)
    z, y, x = np.indices((41, 41, 41), dtype=np.float64)
    center = np.array([20.0, 20.0, 20.0])
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    coords = np.stack((z - center[0], y - center[1], x - center[2]), axis=-1)
    along = coords @ direction
    radial = np.sum(coords**2, axis=-1) - along**2
    label = ((along / 14.0) ** 2 + radial / (3.0**2) <= 1.0).astype(np.uint8)
    intensity = label.astype(np.float32) * 10.0

    oriented_label, _, rotation = pca_orient(label, intensity, spacing)

    pc1 = _principal_axis(oriented_label, spacing)
    assert pc1[0] > 0.95
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_pca_orient_off_center_elongated_object_centers_and_aligns_pc1_with_z() -> None:
    from microct_analysis.processing.orientation import pca_orient

    spacing = (1.0, 1.0, 1.0)
    z, y, x = np.indices((41, 41, 41), dtype=np.float64)
    center = np.array([12.0, 25.0, 14.0])
    direction = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    coords = np.stack((z - center[0], y - center[1], x - center[2]), axis=-1)
    along = coords @ direction
    radial = np.sum(coords**2, axis=-1) - along**2
    label = ((along / 10.0) ** 2 + radial / (2.5**2) <= 1.0).astype(np.uint8)
    intensity = label.astype(np.float32)

    oriented_label, _, _ = pca_orient(label, intensity, spacing)

    pc1 = _principal_axis(oriented_label, spacing)
    assert pc1[0] > 0.95
    np.testing.assert_allclose(_object_center(oriented_label, spacing), np.array([20.0, 20.0, 20.0]), atol=1.0)


def test_apply_rotation_preserves_volume_shape() -> None:
    from microct_analysis.processing.orientation import apply_rotation

    volume = np.zeros((5, 7, 9), dtype=np.float32)
    rotation = np.eye(3)

    rotated = apply_rotation(volume, rotation, order=1, spacing=(0.5, 1.0, 2.0))

    assert rotated.shape == volume.shape


def test_pca_orient_label_mask_uses_nearest_neighbor_without_fractional_labels() -> None:
    from microct_analysis.processing.orientation import pca_orient

    label = np.zeros((21, 21, 21), dtype=np.uint8)
    label[4:17, 9:12, 9:12] = 2
    intensity = label.astype(np.float32)

    oriented_label, _, _ = pca_orient(label, intensity, spacing=(1.0, 1.0, 1.0))

    assert set(np.unique(oriented_label)).issubset({0, 2})


def test_pca_orient_intensity_volume_uses_linear_interpolation() -> None:
    from microct_analysis.processing.orientation import pca_orient

    label = np.zeros((21, 21, 21), dtype=np.uint8)
    label[4:17, 9:12, 9:12] = 1
    intensity = np.zeros_like(label, dtype=np.float32)
    intensity[label == 1] = np.linspace(0.0, 1.0, np.count_nonzero(label))

    _, oriented_intensity, _ = pca_orient(label, intensity, spacing=(1.0, 1.0, 1.0))

    non_binary_values = oriented_intensity[(oriented_intensity > 0.0) & (oriented_intensity < 1.0)]
    assert non_binary_values.size > 0
