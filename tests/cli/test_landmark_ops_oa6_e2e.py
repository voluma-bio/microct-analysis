from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from microct_analysis.processing.session_cache import write_session_cache

from conftest import NOTCH_VOXEL, SPACING, get_landmark, write_json


def test_oa6_landmark_cli_e2e(tmp_path: Path, oa6_femur_context: dict[str, Any]) -> None:
    cache_dir = tmp_path / "cache"
    write_session_cache(
        {
            "labels": None,
            "intensity": None,
            "paths": {"labels": None, "intensity": None},
            "assignments": {"femur": 1},
            "spacing": tuple(SPACING),
            "meshes": {
                "femur": {
                    "render": oa6_femur_context["render_mesh"],
                    "snap_vertices": oa6_femur_context["physical_vertices"],
                    "snap_kdtree": oa6_femur_context["snap_kdtree"],
                }
            },
        },
        cache_dir,
    )

    landmarks_path = write_json(tmp_path / "landmarks.json", {"landmarks": oa6_femur_context["landmarks"]})
    frame = _run_landmark_cli(
        "build-frame",
        "--cache",
        str(cache_dir),
        "--bone",
        "femur",
        "--landmarks",
        str(landmarks_path),
    )
    assert frame["confidence"] in {"high", "medium"}
    assert frame["ap_verification"] in {"mesh_density_confirmed", "recession_differential_confirmed"}

    frame_path = write_json(tmp_path / "frame.json", frame)
    notch_def_path = write_json(
        tmp_path / "notch.json",
        {"domain": "femoral_3d_surface", "id": "intercondylar_notch"},
    )
    notch_coord = (NOTCH_VOXEL * SPACING).tolist()

    accepted = _run_landmark_cli(
        "backstop",
        "--cache",
        str(cache_dir),
        "--bone",
        "femur",
        "--landmark-def",
        str(notch_def_path),
        "--coord",
        _coord_arg(notch_coord),
        "--placed",
        str(landmarks_path),
        "--frame",
        str(frame_path),
    )
    assert accepted["accepted"] is True
    assert accepted["confidence"] == "high"

    swapped = [dict(item) for item in oa6_femur_context["landmarks"]]
    groove = get_landmark(swapped, "intercondylar_groove_midpoint")
    notch = get_landmark(swapped, "intercondylar_notch")
    groove["physical"], notch["physical"] = notch["physical"], groove["physical"]
    groove["voxel"], notch["voxel"] = notch["voxel"], groove["voxel"]
    swapped_path = write_json(tmp_path / "swapped_landmarks.json", {"landmarks": swapped})

    rejected = _run_landmark_cli(
        "backstop",
        "--cache",
        str(cache_dir),
        "--bone",
        "femur",
        "--landmark-def",
        str(notch_def_path),
        "--coord",
        _coord_arg(notch["physical"]),
        "--placed",
        str(swapped_path),
        "--frame",
        str(frame_path),
    )
    assert rejected["accepted"] is False
    assert rejected["signals"]["recession_differential"]["accept"] is False

    workflow_orientation_path = write_json(tmp_path / "workflow_orientation.json", {})
    source_artifacts_path = write_json(tmp_path / "source_artifacts.json", {})
    emit = _run_landmark_cli(
        "emit",
        "--landmarks",
        str(landmarks_path),
        "--workflow-orientation",
        str(workflow_orientation_path),
        "--spacing",
        _coord_arg(SPACING.tolist()),
        "--source-artifacts",
        str(source_artifacts_path),
        "--cache",
        str(cache_dir),
        "--bone",
        "femur",
        "--out-dir",
        str(tmp_path / "emitted"),
    )
    positions_path = Path(emit["artifacts"]["positions"])
    assert positions_path.exists()
    positions = json.loads(positions_path.read_text())
    assert len(positions["landmarks"]) == 4
    assert "_derived_frame" in positions


def test_no_kernel_used() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import microct_analysis.cli.landmark_ops; "
            "print(sorted(m for m in sys.modules if 'kernel' in m or 'jupyter' in m or 'zmq' in m))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    modules = ast.literal_eval(proc.stdout.strip())
    assert not [name for name in modules if name.startswith(("ipykernel", "jupyter_workbench", "zmq"))]
    assert all(name.startswith("jupyter_client") for name in modules), modules


def _run_landmark_cli(*args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "microct_analysis.cli.landmark_ops", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _coord_arg(coord: list[float]) -> str:
    return ",".join(str(float(value)) for value in coord)
