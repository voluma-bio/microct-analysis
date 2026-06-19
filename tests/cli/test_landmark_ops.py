from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from microct_analysis.processing.backstop import compute_backstop
from microct_analysis.processing.session_cache import write_session_cache

from conftest import NOTCH_VOXEL, SPACING, get_landmark, write_json


def test_backstop_cli_matches_in_process_and_rejects_groove_notch_swap(tmp_path: Path, oa6_femur_context: dict[str, Any]) -> None:
    context = oa6_femur_context
    cache_dir = tmp_path / "cache"
    write_session_cache({
        "labels": None,
        "intensity": None,
        "paths": {"labels": None, "intensity": None},
        "assignments": {"femur": 1},
        "spacing": tuple(SPACING),
        "meshes": {"femur": {
            "render": context["render_mesh"],
            "snap_vertices": context["physical_vertices"],
            "snap_kdtree": context["snap_kdtree"],
        }},
    }, cache_dir)

    placed_path = write_json(tmp_path / "placed.json", {"landmarks": context["landmarks"]})
    frame_path = write_json(tmp_path / "frame.json", _frame_payload(context["frame"]))
    landmark_def = write_json(tmp_path / "landmark_def.json", {"domain": "femoral_3d_surface", "id": "intercondylar_notch"})
    notch_coord = (NOTCH_VOXEL * SPACING).tolist()

    cli_result = _run_backstop(cache_dir, landmark_def, notch_coord, placed_path, frame_path)
    assert cli_result["accepted"] is True
    assert cli_result["confidence"] == "high"

    direct = compute_backstop(
        {"domain": "femoral_3d_surface", "id": "intercondylar_notch"},
        tuple(notch_coord),
        mesh_vertices=context["physical_vertices"],
        placed_landmarks={"landmarks": context["landmarks"]},
        femoral_frame=context["frame"],
    )
    direct_dict = asdict(direct)
    assert cli_result["accepted"] == direct_dict["accepted"]
    assert cli_result["confidence"] == direct_dict["confidence"]
    for name, signal in direct_dict["signals"].items():
        assert cli_result["signals"][name]["accept"] == signal["accept"]

    swapped = [dict(item) for item in context["landmarks"]]
    groove = get_landmark(swapped, "intercondylar_groove_midpoint")
    notch = get_landmark(swapped, "intercondylar_notch")
    groove["physical"], notch["physical"] = notch["physical"], groove["physical"]
    groove["voxel"], notch["voxel"] = notch["voxel"], groove["voxel"]
    swapped_path = write_json(tmp_path / "swapped.json", {"landmarks": swapped})

    rejected = _run_backstop(cache_dir, landmark_def, notch["physical"], swapped_path, frame_path)
    assert rejected["accepted"] is False
    assert rejected["signals"]["recession_differential"]["accept"] is False


def test_missing_args_use_json_error_envelope() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "microct_analysis.cli.landmark_ops"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stderr)
    assert "error" in payload


def _run_backstop(cache_dir: Path, landmark_def: Path, coord: list[float], placed: Path, frame: Path) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microct_analysis.cli.landmark_ops",
            "backstop",
            "--cache", str(cache_dir),
            "--landmark-def", str(landmark_def),
            "--coord", ",".join(str(float(v)) for v in coord),
            "--bone", "femur",
            "--placed", str(placed),
            "--frame", str(frame),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _frame_payload(frame) -> dict:
    return {
        "e_ML": frame.e_ML.tolist(),
        "e_AP": frame.e_AP.tolist(),
        "e_SI": frame.e_SI.tolist(),
        "confidence": frame.confidence,
        "evidence": frame.evidence,
    }
