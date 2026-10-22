"""Guard anatomical ROI filling and coordinate registration boundaries."""

import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).parents[1] / "skills/bone-morphometry/resources/python-protocol/iioc.py"
spec = importlib.util.spec_from_file_location("iioc", path)
iioc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iioc)


def test_original_roi_inverse_translation():
    mask = np.zeros((9, 10, 11), bool)
    mask[3, 4, 5] = True
    frame = {"output_xyz_to_original_xyz_matrix": np.eye(3).tolist(), "output_xyz_to_original_xyz_offset": [2, 1, 3]}
    out = iioc.original_roi(mask, frame, (15, 15, 15))
    assert out.sum() == 1
    assert out[6, 5, 7]


def test_growth_face_is_not_forced_flat():
    curves = {"2": [[0, 4], [4, 8]], "6": [[0, 8], [4, 12]]}
    surface = iioc.boundary_surface(curves, (20, 9, 5))
    assert surface[4, 2] == 8
    assert surface[2, 4] - surface[2, 0] == 4


def test_cut_face_fill_retains_marrow_without_filling_outer_notch():
    labels = np.zeros((40, 10, 60), np.uint8)
    # Thick U-shaped shell: top plate, sides; open cut at W25.
    labels[5:15, 2:8, 5:55] = 2
    labels[5:30, 2:8, 5:10] = 2
    labels[5:30, 2:8, 50:55] = 2
    # A true exterior depression remains connected to background.
    labels[5:9, :, 25:35] = 0
    gp = np.full((10, 60), 25.0)
    envelope = iioc.envelope_from_tibia(labels, gp)
    assert envelope[18, 5, 30]  # interior marrow included
    assert not envelope[6, 5, 30]  # exterior not convexified
    roi = envelope & (np.arange(40)[:, None, None] < gp[None])
    assert not roi[25:].any()
