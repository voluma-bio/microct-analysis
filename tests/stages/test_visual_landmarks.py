"""Unit tests for the visual_landmarks stage driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microct_analysis.stages.visual_landmarks import (
    aggregate_confidence,
    emit_positions,
    recommended_action,
)


# --- aggregate_confidence ---


def test_aggregate_confidence_all_high() -> None:
    landmarks = [
        {"id": "intercondylar_groove_midpoint", "confidence": "high"},
        {"id": "lateral_condylar_edge", "confidence": "high"},
        {"id": "medial_condylar_edge", "confidence": "high"},
        {"id": "intercondylar_notch", "confidence": "high"},
    ]
    level, evidence = aggregate_confidence(landmarks)
    assert level == "high"


def test_aggregate_confidence_any_low() -> None:
    landmarks = [
        {"id": "intercondylar_groove_midpoint", "confidence": "high"},
        {"id": "lateral_condylar_edge", "confidence": "low"},
        {"id": "medial_condylar_edge", "confidence": "high"},
        {"id": "intercondylar_notch", "confidence": "high"},
    ]
    level, evidence = aggregate_confidence(landmarks)
    assert level == "low"
    assert "lateral_condylar_edge" in evidence


def test_aggregate_confidence_any_medium() -> None:
    landmarks = [
        {"id": "intercondylar_groove_midpoint", "confidence": "high"},
        {"id": "lateral_condylar_edge", "confidence": "medium"},
        {"id": "medial_condylar_edge", "confidence": "high"},
        {"id": "intercondylar_notch", "confidence": "high"},
    ]
    level, evidence = aggregate_confidence(landmarks)
    assert level == "medium"


def test_aggregate_confidence_iioc_out_of_range() -> None:
    landmarks = [
        {"id": "articular_surface_proximal", "confidence": "high", "voxel": [20.0, 0.0, 0.0]},
        {"id": "growth_plate_proximal", "confidence": "high", "voxel": [145.0, 0.0, 0.0]},
    ]
    level, _evidence = aggregate_confidence(landmarks)
    assert level == "medium"




# --- recommended_action ---


def test_recommended_action_mapping() -> None:
    assert recommended_action("high") == "proceed"
    assert recommended_action("medium") == "flag"
    assert recommended_action("low") == "pause"


# --- emit_positions ---


def test_emit_positions_writes_files(tmp_path: Path) -> None:
    # Create a minimal labels volume
    labels = np.zeros((10, 10, 10), dtype=np.uint8)
    labels[3:7, 3:7, 3:7] = 1
    np.save(tmp_path / "labels.npy", labels)

    source_artifacts = {"labels": str(tmp_path / "labels.npy")}

    workflow_orientation = {
        "target_plane": "frontal",
        "axes": {
            "medial_lateral": {
                "from": "medial_condylar_edge",
                "to": "lateral_condylar_edge",
            },
        },
    }

    placed_landmarks = [
        {
            "id": "intercondylar_groove_midpoint",
            "physical": [50.0, 40.0, 50.0],
            "voxel": [5.0, 4.0, 5.0],
            "confidence": "high",
            "evidence": "surface feature detected",
        },
        {
            "id": "lateral_condylar_edge",
            "physical": [50.0, 40.0, 70.0],
            "voxel": [5.0, 4.0, 7.0],
            "confidence": "high",
            "evidence": "surface feature detected",
        },
        {
            "id": "medial_condylar_edge",
            "physical": [50.0, 40.0, 30.0],
            "voxel": [5.0, 4.0, 3.0],
            "confidence": "high",
            "evidence": "surface feature detected",
        },
        {
            "id": "intercondylar_notch",
            "physical": [55.0, 45.0, 50.0],
            "voxel": [5.5, 4.5, 5.0],
            "confidence": "high",
            "evidence": "surface feature detected",
        },
    ]

    output_dir = tmp_path / "output"
    emit_positions(
        placed_landmarks=placed_landmarks,
        source_artifacts=source_artifacts,
        workflow_orientation=workflow_orientation,
        output_dir=str(output_dir),
    )

    # All expected output files exist
    assert (output_dir / "positions.json").exists()
    assert (output_dir / "orientation_frame.json").exists()
    assert (output_dir / "oriented_labels.npy").exists()
    assert (output_dir / "transform_matrix.json").exists()

    positions = json.loads((output_dir / "positions.json").read_text())
    assert "_derived_frame" not in positions

    # orientation_frame.json has pca_orientation stub with applied=false
    frame = json.loads((output_dir / "orientation_frame.json").read_text())
    assert "pca_orientation" in frame
    assert frame["pca_orientation"]["applied"] is False
