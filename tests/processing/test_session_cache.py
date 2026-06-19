from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from microct_analysis.processing.rendering import prepare_landmark_session
from microct_analysis.processing.session_cache import load_session_cache, write_session_cache


def test_session_cache_round_trip_preserves_meshes_kdtree_and_paths(tmp_path: Path) -> None:
    labels = np.zeros((32, 32, 32), dtype=np.uint8)
    zz, yy, xx = np.indices(labels.shape)
    labels[((zz - 12) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 7**2] = 1
    labels[((zz - 23) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 4**2] = 2

    labels_path = tmp_path / "labels.nii.gz"
    assignments_path = tmp_path / "assignments.json"
    metadata_path = tmp_path / "metadata.json"
    nib.save(nib.Nifti1Image(labels, affine=np.eye(4)), labels_path)
    assignments_path.write_text(json.dumps({"assignments": {"femur": 1}}))
    metadata_path.write_text(json.dumps({"spacing": [0.2, 0.3, 0.4]}))

    session = prepare_landmark_session({
        "labels": str(labels_path),
        "structure_assignments": str(assignments_path),
        "metadata": str(metadata_path),
    })
    session["paths"] = {"labels": str(labels_path), "intensity": None}

    cache_dir = tmp_path / "cache"
    write_session_cache(session, cache_dir)
    loaded = load_session_cache(cache_dir)

    original = session["meshes"]["femur"]
    restored = loaded["meshes"]["femur"]
    assert np.array_equal(restored["render"].points, original["render"].points)
    assert np.array_equal(restored["snap_vertices"], original["snap_vertices"])
    assert np.array_equal(restored["render"].faces, original["render"].faces)

    sample_points = original["snap_vertices"][[0, len(original["snap_vertices"]) // 2, -1]]
    assert np.array_equal(
        restored["snap_kdtree"].query(sample_points)[1],
        original["snap_kdtree"].query(sample_points)[1],
    )
    assert loaded["spacing"] == (0.2, 0.3, 0.4)
    assert loaded["assignments"] == {"femur": 1}
    assert loaded["paths"] == {"labels": str(labels_path), "intensity": None}
    assert "labels_path" not in loaded
    assert "intensity_path" not in loaded
    assert loaded["labels"] is None
    assert loaded["intensity"] is None

def test_session_cache_rewrite_removes_stale_mesh_subdirs(tmp_path: Path) -> None:
    labels = np.zeros((32, 32, 32), dtype=np.uint8)
    zz, yy, xx = np.indices(labels.shape)
    labels[((zz - 12) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 7**2] = 1
    labels[((zz - 23) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 4**2] = 2

    labels_path = tmp_path / "labels.nii.gz"
    assignments_path = tmp_path / "assignments.json"
    nib.save(nib.Nifti1Image(labels, affine=np.eye(4)), labels_path)
    assignments_path.write_text(json.dumps({"assignments": {"femur": 1, "tibia": 2}}))

    session = prepare_landmark_session({
        "labels": str(labels_path),
        "structure_assignments": str(assignments_path),
    })
    session["paths"] = {"labels": str(labels_path), "intensity": None}

    cache_dir = tmp_path / "cache"
    write_session_cache(session, cache_dir)
    assert set(load_session_cache(cache_dir)["meshes"]) == {"femur", "tibia"}

    session["meshes"] = {"femur": session["meshes"]["femur"]}
    session["assignments"] = {"femur": 1}
    write_session_cache(session, cache_dir)

    loaded = load_session_cache(cache_dir)
    assert set(loaded["meshes"]) == {"femur"}
    assert not (cache_dir / "meshes" / "tibia").exists()

