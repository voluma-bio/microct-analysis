import numpy as np
import pytest

from microct_analysis.processing.segmentation import extract_label, seed_from_region, watershed_segment


def test_seed_from_region_flood_fills_masked_components_from_seed_points():
    filtered = np.zeros((1, 5, 7), dtype=np.float32)
    mask = np.zeros_like(filtered, dtype=bool)
    mask[0, 1:4, 1:3] = True
    mask[0, 1:4, 4:6] = True

    seeds = seed_from_region(filtered, mask, {"left": (0, 2, 1), "right": (0, 2, 5)})

    assert seeds.dtype == np.int32
    assert set(np.unique(seeds)) == {0, 1, 2}
    assert np.all(seeds[0, 1:4, 1:3] == 1)
    assert np.all(seeds[0, 1:4, 4:6] == 2)


def test_seed_from_region_rejects_seed_outside_mask():
    filtered = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.zeros_like(filtered, dtype=bool)

    with pytest.raises(ValueError, match="outside mask"):
        seed_from_region(filtered, mask, {"bone": (0, 1, 1)})


def test_watershed_segment_splits_simple_two_region_case():
    filtered = np.array([[[5, 4, 3, 4, 5]]], dtype=np.float32)
    mask = np.ones_like(filtered, dtype=bool)
    seeds = np.zeros_like(filtered, dtype=np.int32)
    seeds[0, 0, 0] = 1
    seeds[0, 0, 4] = 2

    grown = watershed_segment(filtered, seeds, mask)

    assert grown.shape == filtered.shape
    assert set(np.unique(grown)) == {1, 2}
    assert grown[0, 0, 0] == 1
    assert grown[0, 0, 4] == 2


def test_watershed_segment_does_not_label_unseeded_disconnected_mask_islands():
    filtered = np.zeros((1, 1, 5), dtype=np.uint16)
    mask = np.zeros_like(filtered, dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 0, 4] = True
    seeds = np.zeros_like(filtered, dtype=np.int32)
    seeds[0, 0, 0] = 1

    grown = watershed_segment(filtered, seeds, mask)

    assert grown[0, 0, 0] == 1
    assert grown[0, 0, 4] == 0


def test_watershed_segment_handles_unsigned_intensity_without_wraparound():
    filtered = np.array([[[5000, 4000, 3000, 4000, 5000]]], dtype=np.uint16)
    mask = np.ones_like(filtered, dtype=bool)
    seeds = np.zeros_like(filtered, dtype=np.int32)
    seeds[0, 0, 0] = 1
    seeds[0, 0, 4] = 2

    grown = watershed_segment(filtered, seeds, mask)

    assert set(np.unique(grown)) == {1, 2}


def test_extract_label_returns_boolean_mask_for_requested_label():
    grown = np.array([[0, 1, 2], [2, 1, 0]])

    mask = extract_label(grown, 2)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, np.array([[False, False, True], [True, False, False]]))
