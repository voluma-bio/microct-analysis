import numpy as np

from microct_analysis.processing.threshold import apply, binary_mask
from microct_analysis.processing.types import SegmentationThresholds


def test_binary_mask_uses_lower_bound_only():
    volume = np.array([[100, 220, 221]])

    mask = binary_mask(volume, 220)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.array([[False, True, True]]))


def test_binary_mask_uses_inclusive_upper_bound_when_given():
    volume = np.array([[219, 220, 270, 271]])

    mask = binary_mask(volume, 220, 270)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.array([[False, True, True, False]]))


def test_apply_returns_mask_and_opened_marker_volumes():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[1:4, 1:4, 1:4] = 10
    volume[0, 0, 0] = 10
    thresholds = SegmentationThresholds(mask=5, marker=5, method="test")

    mask, markers = apply(volume, thresholds)

    assert mask.dtype == np.uint8
    assert markers.dtype == np.uint8
    assert mask[0, 0, 0] == 1
    assert markers[0, 0, 0] == 0
    assert markers[2, 2, 2] == 1
