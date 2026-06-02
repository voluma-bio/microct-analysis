from __future__ import annotations

import numpy as np
import pytest

from microct_analysis.processing.dicom import LoadError
from microct_analysis.processing import segmentation
from microct_analysis.processing.types import Component, ScanFingerprint, SeedAssignment, Seeds


def _mk_component(index: int, voxel_count: int, centroid_z: float, centroid_y: float = 32.0, centroid_x: float = 32.0) -> Component:
    return Component(
        index=index,
        voxel_count=voxel_count,
        centroid_zyx=(centroid_z, centroid_y, centroid_x),
        centroid_mm=(centroid_z * 0.01, centroid_y * 0.01, centroid_x * 0.01),
        bbox_zyx=((0, 1), (0, 1), (0, 1)),
        edge_faces=(),
    )


def _sphere(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float) -> np.ndarray:
    z, y, x = np.indices(shape, dtype=np.float32)
    cz, cy, cx = center
    return ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius**2


def _four_bone_markers(shape: tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    markers = np.zeros(shape, dtype=np.uint8)
    markers[_sphere(shape, (16, 32, 32), 10)] = 1
    markers[_sphere(shape, (48, 32, 32), 10)] = 1
    markers[_sphere(shape, (30, 12, 32), 4)] = 1
    markers[_sphere(shape, (48, 32, 56), 4)] = 1
    return markers


def test_propose_auto_seeds_is_deterministic_and_tie_breaks_on_index() -> None:
    components = [
        _mk_component(index=7, voxel_count=5000, centroid_z=10.0),
        _mk_component(index=3, voxel_count=5000, centroid_z=12.0),
        _mk_component(index=2, voxel_count=3000, centroid_z=50.0),
    ]

    output = segmentation.propose_auto_seeds(components, z_extent=64)
    reordered = segmentation.propose_auto_seeds(list(reversed(components)), z_extent=64)

    assert output == reordered
    assert set(output) == {"femur", "tibia"}
    assert output["femur"].component_index == 3
    assert output["tibia"].component_index == 2


def test_clean_marker_layout_assigns_canonical_bone_labels() -> None:
    markers = _four_bone_markers()

    result = segmentation.label_bones(markers, spacing=(0.01, 0.01, 0.01), min_marker_voxels=50)

    assert "femur" in result.bone_assignments
    assert "tibia" in result.bone_assignments
    assert "patella" in result.bone_assignments
    assert (result.labeled_markers == 1).sum() > 0
    assert (result.labeled_markers == 2).sum() > 0


def test_too_few_marker_components_escalates() -> None:
    shape = (64, 64, 64)
    markers = np.zeros(shape, dtype=np.uint8)
    markers[_sphere(shape, (16, 32, 32), 10)] = 1
    markers[_sphere(shape, (48, 32, 32), 10)] = 1

    with pytest.raises(LoadError) as excinfo:
        segmentation.label_bones(markers, spacing=(0.01, 0.01, 0.01), min_marker_voxels=50)

    assert excinfo.value.flag == "ambiguous-bone-identity"


def test_adjacent_femur_tibia_markers_escalate_as_bridge() -> None:
    shape = (64, 64, 64)
    markers = np.zeros(shape, dtype=np.uint8)
    markers[_sphere(shape, (20, 32, 32), 10)] = 1
    markers[_sphere(shape, (42, 32, 32), 10)] = 1
    markers[_sphere(shape, (30, 12, 32), 4)] = 1

    with pytest.raises(LoadError) as excinfo:
        segmentation.label_bones(markers, spacing=(0.01, 0.01, 0.01), min_marker_voxels=50)

    assert excinfo.value.flag == "articular-bridging-suspected"


def test_fov_prefilter_drops_thin_edge_sheet() -> None:
    shape = (64, 64, 64)
    markers = np.zeros(shape, dtype=np.uint8)
    markers[28:34, 12:52, 0:2] = 1
    cc, components, _ = segmentation.extract_components(markers, spacing=(0.01, 0.01, 0.01), min_voxels=20)

    retained, dropped = segmentation.fov_prefilter(
        components,
        cc,
        shape,
        segmentation.FovPrefilterConfig(enabled=True, margin_voxels=3),
    )

    assert retained == []
    assert len(dropped) == 1
    assert any("edge-fraction" in reason for reason in dropped[0].reasons)


def _canonical_fingerprint(shape: tuple[int, int, int] = (64, 64, 64)) -> ScanFingerprint:
    return ScanFingerprint(
        slice_uid_hash="abc123",
        volume_shape=shape,
        mask_threshold=2500.0,
        marker_threshold=3500.0,
        min_marker_voxels=50,
        fov_prefilter_enabled=True,
        fov_prefilter_margin=3,
    )


def test_assign_from_seeds_validates_required_bones_and_returns_mapping() -> None:
    markers = _four_bone_markers()
    _, components, _ = segmentation.extract_components(markers, spacing=(0.01, 0.01, 0.01), min_voxels=50)
    fp = _canonical_fingerprint(markers.shape)
    auto = segmentation.propose_auto_seeds(components, z_extent=markers.shape[0])
    seeds = Seeds(schema_version=1, fingerprint=fp, assignments=auto)

    mapping = segmentation.assign_from_seeds(components, seeds, fp)

    assert set(mapping) == {"femur", "tibia"}
    assert mapping["femur"] != mapping["tibia"]


def test_assign_from_seeds_rejects_unsupported_schema_version() -> None:
    markers = _four_bone_markers()
    _, components, _ = segmentation.extract_components(markers, spacing=(0.01, 0.01, 0.01), min_voxels=50)
    fp = _canonical_fingerprint(markers.shape)
    seeds = Seeds(schema_version=999, fingerprint=fp, assignments=segmentation.propose_auto_seeds(components, markers.shape[0]))

    with pytest.raises(LoadError) as excinfo:
        segmentation.assign_from_seeds(components, seeds, fp)

    assert excinfo.value.flag == "invalid-seeds"
    assert "schema_version" in str(excinfo.value)


def test_assign_from_seeds_rejects_drifted_component_stats() -> None:
    markers = _four_bone_markers()
    _, components, _ = segmentation.extract_components(markers, spacing=(0.01, 0.01, 0.01), min_voxels=50)
    fp = _canonical_fingerprint(markers.shape)
    auto = segmentation.propose_auto_seeds(components, z_extent=markers.shape[0])
    femur = auto["femur"]
    auto["femur"] = SeedAssignment(
        component_index=femur.component_index,
        voxel_count=int(femur.voxel_count * 1.5),
        centroid_zyx=femur.centroid_zyx,
    )
    seeds = Seeds(schema_version=1, fingerprint=fp, assignments=auto)

    with pytest.raises(LoadError) as excinfo:
        segmentation.assign_from_seeds(components, seeds, fp)

    assert excinfo.value.flag == "invalid-seeds"
    assert "voxel_count drift" in str(excinfo.value)
