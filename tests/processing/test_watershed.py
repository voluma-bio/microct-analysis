from __future__ import annotations

import numpy as np

from microct_analysis.processing import watershed
from microct_analysis.processing.types import BoneLabels, BoneStats


def test_run_grows_markers_inside_mask_and_reports_stats() -> None:
    filtered = np.zeros((1, 1, 5), dtype=np.float32)
    markers = np.zeros_like(filtered, dtype=np.uint8)
    markers[0, 0, 0] = 1
    markers[0, 0, 4] = 2
    mask = np.ones_like(filtered, dtype=np.uint8)

    result = watershed.run(filtered, markers, mask, spacing=(0.01, 0.01, 0.01))

    assert set(np.unique(result.volume)) == {1, 2}
    assert "femur" in result.per_bone
    assert "tibia" in result.per_bone
    assert result.per_bone["femur"].voxel_count + result.per_bone["tibia"].voxel_count == 5


def test_run_does_not_label_unseeded_disconnected_mask_islands() -> None:
    filtered = np.zeros((1, 1, 5), dtype=np.uint16)
    markers = np.zeros_like(filtered, dtype=np.uint8)
    markers[0, 0, 0] = 1
    mask = np.zeros_like(filtered, dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 0, 4] = 1

    result = watershed.run(filtered, markers, mask, spacing=(0.01, 0.01, 0.01))

    assert result.volume[0, 0, 0] == 1
    assert result.volume[0, 0, 4] == 0


def test_prune_to_seed_cc_removes_label_components_without_seed_connection() -> None:
    labels = np.zeros((1, 1, 5), dtype=np.uint8)
    labels[0, 0, 0:2] = 1
    labels[0, 0, 4] = 1
    markers = np.zeros_like(labels, dtype=np.uint8)
    markers[0, 0, 0] = 1
    bone_labels = BoneLabels(
        volume=labels,
        per_bone={
            "femur": BoneStats(
                label=1,
                name="femur",
                voxel_count=3,
                volume_mm3=3e-6,
                centroid_mm=(0.0, 0.0, 0.0),
                bbox_zyx=((0, 1), (0, 1), (0, 5)),
            )
        },
    )

    cleaned, stats, flags = watershed.prune_to_seed_cc(bone_labels, markers, spacing=(0.01, 0.01, 0.01))

    assert flags == []
    assert stats["femur"] == 1
    assert cleaned.volume[0, 0, 4] == 0
    assert cleaned.volume[0, 0, 0] == 1


def test_prune_to_seed_cc_flags_labels_without_markers() -> None:
    labels = np.zeros((1, 1, 3), dtype=np.uint8)
    labels[0, 0, 0] = 1
    markers = np.zeros_like(labels, dtype=np.uint8)
    bone_labels = BoneLabels(
        volume=labels,
        per_bone={
            "femur": BoneStats(
                label=1,
                name="femur",
                voxel_count=1,
                volume_mm3=1e-6,
                centroid_mm=(0.0, 0.0, 0.0),
                bbox_zyx=((0, 1), (0, 1), (0, 1)),
            )
        },
    )

    _cleaned, stats, flags = watershed.prune_to_seed_cc(bone_labels, markers, spacing=(0.01, 0.01, 0.01))

    assert stats["femur"] == 1
    assert flags == ["pruning-dropped-bone:femur"]
