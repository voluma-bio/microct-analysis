import numpy as np

from microct_analysis.processing.sanity import (
    check,
    check_bone_volume_ordering,
    check_condyle_separation,
    check_femoral_length_plausibility,
    check_iioc_slice_count,
    check_tibial_ratio,
)
from microct_analysis.processing.types import BoneLabels, BoneStats, LabelVolume


def _labels(counts: dict[str, int]) -> LabelVolume:
    label_map = {"femur": 1, "tibia": 2, "fibula": 3, "patella": 4}
    data = np.concatenate([np.full(counts[name], label_id, dtype=np.uint8) for name, label_id in label_map.items()])
    return LabelVolume(data=data, spacing=(0.01, 0.01, 0.01), label_map=label_map)


def _stats(label: int, name: str, voxels: int) -> BoneStats:
    return BoneStats(
        label=label,
        name=name,
        voxel_count=voxels,
        volume_mm3=voxels * 1e-6,
        centroid_mm=(0.0, 0.0, 0.0),
        bbox_zyx=((0, 1), (0, 1), (0, 1)),
    )


def test_bone_volume_ordering_accepts_expected_order():
    labels = _labels({"femur": 10, "tibia": 8, "fibula": 4, "patella": 2})

    assert check_bone_volume_ordering(labels) == []


def test_bone_volume_ordering_warns_on_incorrect_order():
    labels = _labels({"femur": 6, "tibia": 8, "fibula": 4, "patella": 2})

    warnings = check_bone_volume_ordering(labels)

    assert warnings
    assert "femur" in warnings[0]
    assert "tibia" in warnings[0]


def test_bone_volume_ordering_warns_on_missing_label():
    labels = LabelVolume(data=np.array([1, 2, 3], dtype=np.uint8), spacing=(1.0, 1.0, 1.0), label_map={"femur": 1})

    warnings = check_bone_volume_ordering(labels)

    assert warnings
    assert "Missing labels" in warnings[0]


def test_condyle_separation_accepts_two_lobes_with_central_gap():
    rng = np.random.default_rng(0)
    left = np.column_stack([rng.normal(0, 0.1, 80), rng.normal(0, 0.1, 80), rng.normal(-1.0, 0.08, 80)])
    right = np.column_stack([rng.normal(0, 0.1, 80), rng.normal(0, 0.1, 80), rng.normal(1.0, 0.08, 80)])

    assert check_condyle_separation(np.vstack([left, right])) == []


def test_condyle_separation_warns_when_gap_not_detectable():
    rng = np.random.default_rng(1)
    vertices = np.column_stack([rng.normal(0, 0.1, 120), rng.normal(0, 0.1, 120), rng.normal(0.0, 0.2, 120)])

    assert check_condyle_separation(vertices)


def test_femoral_length_boundaries_are_inclusive():
    assert check_femoral_length_plausibility(2.0) == []
    assert check_femoral_length_plausibility(2.6) == []
    assert check_femoral_length_plausibility(1.99)
    assert check_femoral_length_plausibility(2.61)


def test_iioc_slice_count_boundaries_are_inclusive():
    assert check_iioc_slice_count(50) == []
    assert check_iioc_slice_count(100) == []
    assert check_iioc_slice_count(49)
    assert check_iioc_slice_count(101)


def test_tibial_ratio_boundaries_are_inclusive():
    assert check_tibial_ratio(0.15) == []
    assert check_tibial_ratio(0.45) == []
    assert check_tibial_ratio(0.149)
    assert check_tibial_ratio(0.451)


def test_structural_sanity_check_flags_edges_and_volume_order():
    volume = np.zeros((40, 40, 40), dtype=np.uint8)
    volume[20, 0, 20] = 1
    volume[21, 20, 20] = 2
    bone_labels = BoneLabels(
        volume=volume,
        per_bone={
            "femur": _stats(1, "femur", 100),
            "tibia": _stats(2, "tibia", 500),
        },
    )

    flags = check(bone_labels)

    assert "bone-volume-order-wrong" in flags
    assert "bone-on-ap-edge" in flags


def test_structural_sanity_check_flags_component_count_extremes():
    empty = BoneLabels(volume=np.zeros((4, 4, 4), dtype=np.uint8), per_bone={})
    one = BoneLabels(volume=np.zeros((4, 4, 4), dtype=np.uint8), per_bone={"femur": _stats(1, "femur", 1)})

    assert "no-bones-labeled" in check(empty)
    assert "only-one-bone-labeled" in check(one)
