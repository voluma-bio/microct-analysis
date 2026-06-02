import numpy as np
import pytest

from microct_analysis.processing.dicom import LoadError
from microct_analysis.processing.resample import anisotropy_factor, is_isotropic, to_isotropic


def test_isotropic_no_op_returns_input_by_identity() -> None:
    volume = np.ones((2, 3, 4), dtype=np.float32)

    resampled, new_spacing = to_isotropic(volume, (0.01, 0.01, 0.01))

    assert resampled is volume
    assert new_spacing == (0.01, 0.01, 0.01)


def test_isotropic_output_has_equal_spacing_in_all_dimensions() -> None:
    volume = np.ones((2, 3, 4), dtype=np.float32)

    _resampled, new_spacing = to_isotropic(volume, (0.02, 0.01, 0.01))

    assert new_spacing == (0.01, 0.01, 0.01)


def test_anisotropic_input_resamples_axis_by_spacing_ratio() -> None:
    volume = np.ones((2, 3, 4), dtype=np.float32)

    resampled, new_spacing = to_isotropic(volume, (0.02, 0.01, 0.01), order=1)

    assert resampled.shape == (4, 3, 4)
    assert new_spacing == (0.01, 0.01, 0.01)


def test_nearest_neighbor_preserves_discrete_labels() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.uint8)
    labels[1, :, :] = 2

    resampled, _new_spacing = to_isotropic(labels, (0.02, 0.01, 0.01), order=0)

    assert set(np.unique(resampled).tolist()) <= {0, 2}


def test_target_spacing_parameter_controls_output_spacing() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)

    resampled, new_spacing = to_isotropic(volume, (0.02, 0.01, 0.01), target_spacing=0.02)

    assert new_spacing == (0.02, 0.02, 0.02)
    assert resampled.shape == (4, 2, 2)


def test_output_shape_scales_with_zoom_factors() -> None:
    volume = np.ones((3, 4, 5), dtype=np.float32)

    resampled, _new_spacing = to_isotropic(volume, (0.03, 0.02, 0.01), target_spacing=0.01)

    assert resampled.shape == (9, 8, 5)


def test_isotropy_helpers_report_spacing_relationships() -> None:
    assert is_isotropic((0.01, 0.01005, 0.01))
    assert not is_isotropic((0.02, 0.01, 0.01))
    assert anisotropy_factor((0.03, 0.01, 0.02)) == pytest.approx(3.0)


def test_higher_order_interpolation_values_remain_supported() -> None:
    volume = np.ones((2, 3, 4), dtype=np.float32)

    resampled, new_spacing = to_isotropic(volume, (0.02, 0.01, 0.01), order=5)

    assert resampled.shape == (4, 3, 4)
    assert new_spacing == (0.01, 0.01, 0.01)


def test_anisotropy_too_large_escalates() -> None:
    volume = np.zeros((5, 100, 100), dtype=np.float32)

    with pytest.raises(LoadError) as excinfo:
        to_isotropic(volume, (0.10, 0.01, 0.01), anisotropy_limit=3.0)

    assert excinfo.value.flag == "anisotropy-too-large"
