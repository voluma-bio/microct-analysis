"""Sanity checks for segmentation and measurement outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from microct_analysis.processing.types import BoneLabels, LabelVolume, LoadedScan

EXPECTED_ORDER = ("femur", "tibia", "fibula", "patella")


def check(bone_labels: BoneLabels, scan: LoadedScan | None = None) -> list[str]:
    """Run mouse-ct-style structural checks and return flag strings.

    ``scan`` is accepted for compatibility with the source primitive; current
    checks are scanner-invariant and only use the labeled volume/stats.
    """

    del scan
    flags: list[str] = []
    per_bone = bone_labels.per_bone

    present = [name for name in EXPECTED_ORDER if name in per_bone]
    volumes = [per_bone[name].volume_mm3 for name in present]
    for index in range(len(volumes) - 1):
        if volumes[index] < volumes[index + 1]:
            flags.append("bone-volume-order-wrong")
            break

    if len(per_bone) == 0:
        flags.append("no-bones-labeled")
    elif len(per_bone) == 1:
        flags.append("only-one-bone-labeled")
    elif len(per_bone) > 10:
        flags.append("too-many-components")

    volume = np.asarray(bone_labels.volume)
    if volume.ndim != 3:
        return flags
    if (volume[:, 0, :] > 0).any() or (volume[:, -1, :] > 0).any():
        flags.append("bone-on-ap-edge")
    if (volume[:, :, 0] > 0).any() or (volume[:, :, -1] > 0).any():
        flags.append("bone-on-lateral-edge")

    return flags


def check_bone_volume_ordering(labels: LabelVolume) -> list[str]:
    """Verify expected femur > tibia > fibula > patella voxel-count ordering."""

    expected = ["femur", "tibia", "fibula", "patella"]
    missing = [name for name in expected if name not in labels.label_map]
    if missing:
        return [f"Missing labels for bone volume ordering: {', '.join(missing)}"]

    counts = {name: int(np.count_nonzero(labels.data == labels.label_map[name])) for name in expected}
    warnings: list[str] = []
    for larger, smaller in zip(expected, expected[1:]):
        if counts[larger] <= counts[smaller]:
            warnings.append(
                f"Unexpected bone volume ordering: {larger} has {counts[larger]} voxels, "
                f"not greater than {smaller} with {counts[smaller]} voxels."
            )
    return warnings


def check_condyle_separation(femur_mesh_vertices: NDArray[np.floating]) -> list[str]:
    """Check that femoral condyles appear as two separated medial-lateral lobes."""

    vertices = np.asarray(femur_mesh_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 10:
        return ["Femur mesh has too few valid vertices to assess condyle separation."]

    ml = vertices[:, 2]
    ml_range = float(np.ptp(ml))
    if ml_range == 0.0:
        return ["Femur mesh has no medial-lateral width; condyle separation is not detectable."]

    hist, _ = np.histogram(ml, bins=32)
    left_peak = int(np.max(hist[:12]))
    center_valley = int(np.min(hist[12:20]))
    right_peak = int(np.max(hist[20:]))
    weaker_peak = min(left_peak, right_peak)
    if weaker_peak == 0 or center_valley > weaker_peak * 0.6:
        return ["Femoral condyles are not clearly separated by a detectable central gap."]
    return []


def check_femoral_length_plausibility(length_mm: float) -> list[str]:
    """Femoral length must be 2.0-2.6 mm."""

    if 2.0 <= length_mm <= 2.6:
        return []
    return [f"Femoral length {length_mm:.3g} mm is outside plausible range 2.0-2.6 mm."]


def check_iioc_slice_count(n_slices: int) -> list[str]:
    """IIOC slice count must be 50-100."""

    if 50 <= n_slices <= 100:
        return []
    return [f"IIOC slice count {n_slices} is outside expected range 50-100."]


def check_tibial_ratio(ratio: float) -> list[str]:
    """IIOC H/W ratio must be 0.15-0.45."""

    if 0.15 <= ratio <= 0.45:
        return []
    return [f"Tibial IIOC H/W ratio {ratio:.3g} is outside expected range 0.15-0.45."]
