"""Visual landmark stage driver — artifact emission and confidence aggregation.

The landmarker agent drives the placement loop; this module provides
the emit/frame/confidence functions called via jupyter-workbench exec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from microct_analysis.stages.landmarks_orientation import compute_orientation_frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_positions(
    placed_landmarks: list[dict[str, Any]],
    workflow_orientation: dict[str, Any],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    source_artifacts: dict[str, str] | None = None,
    output_dir: str = "landmarks",
) -> dict[str, Any]:
    """Write positions.json, orientation_frame.json, oriented_labels.npy, transform_matrix.json.

    Returns a stage report dict suitable for the analyst.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- positions.json ---
    derived_frame = derive_ml_vector(placed_landmarks)
    positions: dict[str, Any] = {
        "landmarks": placed_landmarks,
        "coordinate_system": "volume_zyx",
        "spacing": list(spacing),
        "source_artifacts": dict(source_artifacts) if source_artifacts else {},
        "orientation_applied": False,
        "placement_method": "visual",
    }
    if derived_frame is not None:
        positions["_derived_frame"] = derived_frame

    _write_json(output_root / "positions.json", positions)

    # --- orientation_frame.json ---
    orientation_frame = compute_orientation_frame(placed_landmarks, workflow_orientation)
    pca_stub = {
        "type": "not-applied",
        "applied": False,
        "confidence": "n/a",
        "label": None,
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation": [0.0, 0.0, 0.0],
        "preserve_voxel_size": True,
        "label_interpolation_order": 0,
        "intensity_interpolation_order": 1,
        "validation": "PCA orientation not used; frame derived from placed landmarks.",
        "explanation": "PCA orientation not used; frame derived from placed landmarks.",
    }
    orientation_frame["pca_orientation"] = pca_stub
    orientation_frame.setdefault("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    orientation_frame.setdefault("translation", [0.0, 0.0, 0.0])
    orientation_frame.setdefault("orientation_confidence", "n/a")
    _write_json(output_root / "orientation_frame.json", orientation_frame)

    # --- transform_matrix.json ---
    _write_json(output_root / "transform_matrix.json", pca_stub)

    # --- oriented_labels.npy (identity copy) ---
    labels = _load_labels(source_artifacts.get("labels"))
    np.save(
        output_root / "oriented_labels.npy",
        labels if labels is not None else np.zeros((0, 0, 0), dtype=np.uint8),
    )

    # --- stage report ---
    confidence, evidence = aggregate_confidence(placed_landmarks)
    return {
        "stage": "landmarks",
        "confidence": confidence,
        "evidence": evidence,
        "recommended_action": recommended_action(confidence),
        "artifacts": {
            "positions": str(output_root / "positions.json"),
            "orientation_frame": str(output_root / "orientation_frame.json"),
            "transform_matrix": str(output_root / "transform_matrix.json"),
            "oriented_labels": str(output_root / "oriented_labels.npy"),
            "screenshots": [],
        },
    }


def aggregate_confidence(
    placed_landmarks: list[dict[str, Any]],
) -> tuple[str, str]:
    """Compute stage-level confidence from per-landmark confidences.

    Rules:
    - ANY landmark at "low" -> stage "low"
    - ANY landmark at "medium" -> stage "medium"
    - ALL "high" -> stage "high"
    - IIOC interval out of range [1, 100] downgrades to "medium"
    """
    low_ids = [lm["id"] for lm in placed_landmarks if lm.get("confidence") == "low"]
    if low_ids:
        return "low", f"Low-confidence landmarks: {', '.join(low_ids)}"

    # IIOC interval check
    iioc_issue = _check_iioc_interval(placed_landmarks)
    if iioc_issue:
        return "medium", iioc_issue

    medium_ids = [lm["id"] for lm in placed_landmarks if lm.get("confidence") == "medium"]
    if medium_ids:
        return "medium", f"Medium-confidence landmarks: {', '.join(medium_ids)}"

    return "high", "All landmarks placed with high confidence via visual placement."


def derive_ml_vector(
    placed_landmarks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compute ML vector from groove, lateral, and medial condylar edge landmarks.

    Returns dict with "ml_vector" (ZYX) and "source_landmarks", or None if
    any of the 3 required landmarks is missing.
    """
    by_id = {lm["id"]: lm for lm in placed_landmarks}

    groove = by_id.get("intercondylar_groove_midpoint")
    lateral = by_id.get("lateral_condylar_edge")
    medial = by_id.get("medial_condylar_edge")

    if groove is None or lateral is None or medial is None:
        return None

    lat_phys = np.asarray(lateral["physical"], dtype=float)
    med_phys = np.asarray(medial["physical"], dtype=float)
    ml_raw = lat_phys - med_phys
    norm = float(np.linalg.norm(ml_raw))
    if norm == 0:
        return None
    ml_unit = (ml_raw / norm).tolist()

    return {
        "ml_vector": ml_unit,
        "source_landmarks": [groove["id"], lateral["id"], medial["id"]],
    }


def recommended_action(confidence: str) -> str:
    """Map confidence to recommended action."""
    return {"high": "proceed", "medium": "flag", "low": "pause"}.get(confidence, "pause")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_iioc_interval(placed_landmarks: list[dict[str, Any]]) -> str | None:
    """Check growth plate to articular surface interval is in [1, 100] slices."""
    articular = None
    growth = None
    for lm in placed_landmarks:
        lm_id = str(lm.get("id", ""))
        if "articular" in lm_id:
            articular = lm
        elif "growth_plate" in lm_id:
            growth = lm

    if articular is None or growth is None:
        return None

    art_z = float(articular.get("voxel", [0.0])[0])
    gp_z = float(growth.get("voxel", [0.0])[0])
    iioc_slices = gp_z - art_z

    if iioc_slices <= 0 or iioc_slices > 100:
        return (
            f"IIOC interval implausible: {iioc_slices:.0f} slices "
            f"(articular={art_z:.0f}, growth_plate={gp_z:.0f}). "
            "Expected 1-100 slices."
        )
    return None


def _load_labels(path: str | None) -> np.ndarray | None:
    """Load label volume from path."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    suffix = "".join(p.suffixes).lower()
    if suffix == ".npy":
        return np.load(p)
    if ".nii" in suffix:
        import nibabel as nib

        return np.asarray(nib.load(str(p)).dataobj)
    return None


def _write_json(path: Path, data: Any) -> None:
    """Write JSON with indent=2."""
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
