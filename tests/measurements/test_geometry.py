import pytest
import numpy as np

from microct_analysis.measurements.geometry import compute_distance, compute_ratio, compute_slice_count, compute_frontal_projected_width
from microct_analysis.measurements.models import MeasurementSpec


def test_distance_uses_spacing():
    spec = MeasurementSpec(name="length", kind="distance", points=["a", "b"])
    result = compute_distance(spec, {"a": [0, 0, 0], "b": [3, 4, 0]}, (2.0, 1.0, 1.0))
    assert result.value == pytest.approx((6**2 + 4**2) ** 0.5)


def test_ratio_uses_component_results():
    a = compute_distance(MeasurementSpec("a", "distance", points=["p1", "p2"]), {"p1": [0, 0, 0], "p2": [2, 0, 0]}, (1, 1, 1))
    b = compute_distance(MeasurementSpec("b", "distance", points=["p1", "p2"]), {"p1": [0, 0, 0], "p2": [4, 0, 0]}, (1, 1, 1))
    result = compute_ratio(MeasurementSpec("r", "ratio", numerator="b", denominator="a", unit="dimensionless"), {"a": a, "b": b})
    assert result.value == pytest.approx(2.0)


def test_slice_count_uses_dominant_axis_spacing():
    spec = MeasurementSpec(name="height", kind="slice_count", boundaries=["top", "bottom"])
    result = compute_slice_count(spec, {"top": [2, 0, 0], "bottom": [7, 1, 0]}, (0.01, 1, 1))
    assert result.value == pytest.approx(0.05)


def test_frontal_projected_width_uses_derived_ml_vector():
    spec = MeasurementSpec(
        name="femoral_width",
        kind="frontal_projected_width",
        points=["lateral", "medial"],
        projection="frontal",
    )
    landmarks = {
        "lateral": [10.0, 5.0, 8.0],
        "medial": [10.0, 5.0, 2.0],
        "_derived_frame": {
            "ml_vector": [0.0, 0.0, 1.0],
            "source_landmarks": ["intercondylar_groove_midpoint", "lateral_condylar_edge", "medial_condylar_edge"],
        },
    }
    spacing = (0.0105, 0.0105, 0.0105)
    result = compute_frontal_projected_width(spec, landmarks, spacing)
    assert result.value == pytest.approx(6.0 * 0.0105)


def test_frontal_projected_width_rotated_ml_vector():
    spec = MeasurementSpec(
        name="femoral_width",
        kind="frontal_projected_width",
        points=["lateral", "medial"],
        projection="frontal",
    )
    ml_vec = [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
    landmarks = {
        "lateral": [10.0, 5.0, 8.0],
        "medial": [10.0, 5.0, 2.0],
        "_derived_frame": {"ml_vector": ml_vec},
    }
    spacing = (0.0105, 0.0105, 0.0105)
    result = compute_frontal_projected_width(spec, landmarks, spacing)
    expected = 6.0 * 0.0105 * (1.0 / np.sqrt(2))
    assert result.value == pytest.approx(expected)


def test_frontal_projected_width_raises_without_derived_frame():
    spec = MeasurementSpec(
        name="femoral_width",
        kind="frontal_projected_width",
        points=["lateral", "medial"],
        projection="frontal",
    )
    landmarks = {
        "lateral": [10.0, 5.0, 8.0],
        "medial": [10.0, 5.0, 2.0],
    }
    with pytest.raises(ValueError, match="landmark-derived ML vector required"):
        compute_frontal_projected_width(spec, landmarks, (0.0105, 0.0105, 0.0105))
