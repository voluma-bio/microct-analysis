from __future__ import annotations

import pytest

from microct_analysis.domain.artifact_contracts import IntakeArtifacts, SegmentationArtifacts, screenshot_path


def test_intake_artifacts_include_stage_report_without_moving_existing_paths() -> None:
    artifacts = IntakeArtifacts()

    assert artifacts.volume_metadata == "intake/volume_metadata.json"
    assert artifacts.orientation_report == "intake/orientation_report.md"
    assert artifacts.stage_report == "intake/stage_report.json"


def test_segmentation_artifacts_include_full_stage_contract() -> None:
    artifacts = SegmentationArtifacts()

    assert artifacts.labels == "segmentation/labels.nii.gz"
    assert artifacts.structure_assignments == "segmentation/structure_assignments.json"
    assert artifacts.seeds == "segmentation/seeds.json"
    assert artifacts.metadata == "segmentation/metadata.json"
    assert artifacts.stage_report == "segmentation/stage_report.json"
    assert artifacts.components == "segmentation/components.nii.gz"
    assert artifacts.masks_dir == "segmentation/masks"


def test_screenshot_path_zero_pads_index() -> None:
    assert screenshot_path("segmentation", 1) == "segmentation/screenshot_001.png"
    assert screenshot_path("measurements", 12) == "measurements/screenshot_012.png"


def test_screenshot_path_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        screenshot_path("", 1)
    with pytest.raises(ValueError):
        screenshot_path("segmentation/nested", 1)
    with pytest.raises(ValueError):
        screenshot_path("segmentation", 0)
