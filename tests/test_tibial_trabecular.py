"""Sphere-map checks against known plates and independent discrete balls."""

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy import ndimage as ndi

root = Path(__file__).parents[1] / "skills/bone-morphometry/resources/python-protocol"
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("tibial_trabecular", root / "tibial_trabecular.py")
trab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trab)


def test_even_plate_has_full_local_thickness_at_surface_and_center():
    phase = np.zeros((20, 30, 30), bool)
    phase[7:13, 2:28, 2:28] = True
    diameter = trab.local_thickness(phase)
    assert np.all(diameter[7:13, 10:20, 10:20] == 6)
    # Mean twice-EDT would incorrectly report4 voxels for this plate interior.
    assert np.mean(2 * ndi.distance_transform_edt(phase)[7:13, 10:20, 10:20]) == 4
    assert not diameter[~phase].any()


def test_sphere_propagation_matches_independent_enumeration():
    phase = np.zeros((9, 10, 11), bool)
    phase[2:7, 2:8, 2:9] = True
    phase[2:5, 2:5, 2:5] = False
    measured = trab.local_thickness(phase)
    foreground = np.argwhere(phase)
    background = np.argwhere(~phase)
    expected = np.zeros_like(measured)
    for center in foreground:
        nearest = np.sqrt(np.sum((background - center) ** 2, axis=1)).min()
        radius = np.floor(nearest * 2) / 2
        contained = np.sum((foreground - center) ** 2, axis=1) < radius**2
        points = foreground[contained]
        expected[tuple(points.T)] = np.maximum(expected[tuple(points.T)], 2 * radius)
    np.testing.assert_array_equal(measured, expected)


def test_empty_phase_is_not_reported_as_zero_thickness():
    with pytest.raises(ValueError, match="Empty phase"):
        trab.local_thickness(np.zeros((4, 4, 4), bool))
