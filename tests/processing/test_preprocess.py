import numpy as np
import pytest

from microct_analysis.processing.preprocess import median_filter


def test_median_filter_reduces_xy_slice_noise():
    volume = np.zeros((1, 5, 5), dtype=np.float32)
    volume[0, 2, 2] = 99

    filtered = median_filter(volume, iterations=1, size=3)

    assert filtered[0, 2, 2] == 0
    assert filtered.shape == volume.shape


def test_median_filter_iterations_parameter_applies_repeatedly():
    volume = np.zeros((1, 7, 7), dtype=np.float32)
    volume[0, 2:5, 2:5] = 10
    volume[0, 3, 3] = 0

    once = median_filter(volume, iterations=1, size=3)
    twice = median_filter(volume, iterations=2, size=3)

    assert once[0, 3, 3] == 10
    assert np.count_nonzero(twice) < np.count_nonzero(once)


def test_median_filter_is_slice_by_slice_not_across_z():
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[0, :, :] = 9
    volume[1, 1, 1] = 1
    volume[2, :, :] = 9

    filtered = median_filter(volume, iterations=1, size=3)

    assert filtered[1, 1, 1] == 0


def test_median_filter_zero_iterations_preserves_dtype_and_returns_copy():
    volume = np.arange(8, dtype=np.uint16).reshape((2, 2, 2))

    filtered = median_filter(volume, iterations=0)

    assert filtered.dtype == volume.dtype
    assert filtered is not volume
    np.testing.assert_array_equal(filtered, volume)


def test_median_filter_radius_uses_simpleitk_compatibility_path():
    volume = np.zeros((1, 5, 5), dtype=np.uint16)
    volume[0, 2, 2] = 99

    filtered = median_filter(volume, iterations=1, radius=1)

    assert filtered.dtype == np.float32
    assert filtered[0, 2, 2] == 0


def test_median_filter_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="iterations"):
        median_filter(np.zeros((1, 1, 1)), iterations=-1)

    with pytest.raises(ValueError, match="size"):
        median_filter(np.zeros((1, 1, 1)), size=0)

    with pytest.raises(ValueError, match="radius"):
        median_filter(np.zeros((1, 1, 1)), radius=-1)
