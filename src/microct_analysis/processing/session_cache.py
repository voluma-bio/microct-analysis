"""Disk cache for visual-landmark session state.

The cache intentionally stores mesh-derived state but not full label/intensity
volumes.  ``load_session_cache`` returns ``labels`` and ``intensity`` as
``None`` and exposes volume paths under ``paths``; callers that need pixels
should explicitly call ``load_volume(path)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree


def write_session_cache(session: dict[str, Any], cache_dir: str | Path) -> None:
    """Persist a landmark session to ``cache_dir`` without storing volumes."""
    root = Path(cache_dir)
    meshes_root = root / "meshes"
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(meshes_root, ignore_errors=True)
    meshes_root.mkdir(parents=True, exist_ok=True)

    paths = dict(session.get("paths") or {})
    labels_path = paths.get("labels")
    intensity_path = paths.get("intensity")

    payload = {
        "spacing": list(session.get("spacing", (1.0, 1.0, 1.0))),
        "assignments": {str(k): int(v) for k, v in dict(session.get("assignments", {})).items()},
        "paths": {"labels": str(labels_path) if labels_path is not None else None,
                  "intensity": str(intensity_path) if intensity_path is not None else None},
    }
    (root / "session.json").write_text(json.dumps(payload, indent=2) + "\n")

    for bone_name, mesh_data in dict(session.get("meshes", {})).items():
        bone_dir = meshes_root / str(bone_name)
        bone_dir.mkdir(parents=True, exist_ok=True)
        render_mesh = mesh_data["render"]
        np.savez_compressed(bone_dir / "render_verts.npz", data=np.asarray(render_mesh.points))
        np.savez_compressed(bone_dir / "render_faces.npz", data=np.asarray(render_mesh.faces))
        np.savez_compressed(bone_dir / "snap_vertices.npz", data=np.asarray(mesh_data["snap_vertices"]))


def load_session_cache(cache_dir: str | Path) -> dict[str, Any]:
    """Load cached meshes/KDTrees and volume paths.

    Full label/intensity volumes are not loaded eagerly.  The returned dict has
    ``labels`` and ``intensity`` set to ``None`` plus ``paths``.  Use
    ``load_volume(path)`` in slice or tibial backstop callers when pixels are
    actually needed.
    """
    import pyvista as pv

    root = Path(cache_dir)
    payload = json.loads((root / "session.json").read_text())
    paths = payload.get("paths") or {}
    meshes: dict[str, dict[str, Any]] = {}

    for bone_dir in sorted((root / "meshes").iterdir()):
        if not bone_dir.is_dir():
            continue
        render_verts = _load_npz_array(bone_dir / "render_verts.npz")
        render_faces = _load_npz_array(bone_dir / "render_faces.npz")
        snap_vertices = _load_npz_array(bone_dir / "snap_vertices.npz")
        meshes[bone_dir.name] = {
            "render": pv.PolyData(render_verts, render_faces),
            "snap_vertices": snap_vertices,
            "snap_kdtree": KDTree(snap_vertices),
        }

    return {
        "labels": None,
        "intensity": None,
        "paths": {"labels": paths.get("labels"), "intensity": paths.get("intensity")},
        "assignments": {str(k): int(v) for k, v in dict(payload.get("assignments", {})).items()},
        "spacing": tuple(float(v) for v in payload.get("spacing", (1.0, 1.0, 1.0))),
        "meshes": meshes,
    }


def load_volume(path: str | Path | None) -> np.ndarray:
    """Load a NIfTI (.nii/.nii.gz) or .npy volume from ``path``."""
    if path is None:
        raise ValueError("volume path is not available in the session cache")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    suffix = "".join(p.suffixes).lower()
    if ".nii" in suffix:
        import nibabel as nib

        return np.asarray(nib.load(str(p)).dataobj)
    if suffix == ".npy":
        return np.load(str(p))
    raise ValueError(f"unsupported file format: {suffix}")


def _load_npz_array(path: Path) -> np.ndarray:
    with np.load(path, mmap_mode="r") as payload:
        if "data" in payload:
            return np.asarray(payload["data"])
        return np.asarray(payload[payload.files[0]])
