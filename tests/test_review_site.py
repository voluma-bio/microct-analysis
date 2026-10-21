"""Protect the image/overlay orientation and invalid-request boundary."""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SOURCE = Path(__file__).parents[1] / "skills/bone-morphometry/resources/review-site"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("microct_review_server", SOURCE / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
sys.path.remove(str(SOURCE))


def test_planes_keep_array_row_column_order_and_overlay_only_labels():
    hu = np.arange(24).reshape(2, 3, 4).astype(float)
    labels = np.zeros(hu.shape, dtype=np.uint8)
    labels[1, 1, 2] = 1
    for axis in range(3):
        index = 1
        raw = np.array(Image.open(io.BytesIO(server.slice_png(hu, labels, axis, index, 0, 23, False))))
        expected = (np.take(hu, index, axis=axis) / 23 * 255).astype(np.uint8)
        np.testing.assert_array_equal(raw[..., 0], expected)
        overlay = np.array(Image.open(io.BytesIO(server.slice_png(hu, labels, axis, index, 0, 23, True))))
        mask = np.take(labels, index, axis=axis) > 0
        np.testing.assert_array_equal(raw[~mask], overlay[~mask])
        if mask.any():
            assert np.any(raw[mask] != overlay[mask])


@pytest.mark.parametrize(
    "axis,index,low,high", [(3, 0, 0, 10), (0, -1, 0, 10), (0, 2, 0, 10), (0, 0, 10, 10), (0, 0, 20, 10)]
)
def test_reject_invalid_slice_or_window(axis, index, low, high):
    hu = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        server.slice_png(hu, hu.astype(np.uint8), axis, index, low, high, True)
