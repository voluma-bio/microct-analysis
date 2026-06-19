"""Stateless CLI wrappers for visual-landmark primitives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


class JsonErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(json.dumps({"error": message}), file=sys.stderr)
        sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("PYVISTA_USE_PANEL", "false")

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI contract requires JSON errors for all failures.
        print(json.dumps({"error": str(exc), "trace": repr(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, default=_json_default))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = JsonErrorArgumentParser(prog="mct-landmark")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=JsonErrorArgumentParser)

    p = sub.add_parser("prepare")
    p.add_argument("--seg", required=True)
    p.add_argument("--cache", required=True)
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser(
        "render-surface",
        description="Render using camera JSON position/focal_point/view_up/view_angle; set view_angle instead of zoom.",
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_render_surface)

    p = sub.add_parser("render-slice")
    p.add_argument("--cache", required=True)
    p.add_argument("--volume", choices=("labels", "intensity"), required=True)
    p.add_argument("--axis", type=int, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--window", type=_float_pair, required=True)
    p.add_argument("--mask-bone")
    p.add_argument("--resolution", type=_int_pair, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_render_slice)

    p = sub.add_parser("snap-surface")
    p.add_argument("--cache", required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--pixel", type=_int_pair, required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--resolution", type=_int_pair)
    p.set_defaults(func=_cmd_snap_surface)

    p = sub.add_parser("snap-slice")
    p.add_argument("--cache", required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--slice", type=int, required=True)
    p.add_argument("--pixel", type=_int_pair, required=True)
    p.add_argument("--mode", choices=("center", "edge_medial", "edge_lateral", "exact"), required=True)
    p.set_defaults(func=_cmd_snap_slice)

    p = sub.add_parser("build-frame")
    p.add_argument("--landmarks", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--bone", required=True)
    p.set_defaults(func=_cmd_build_frame)

    p = sub.add_parser("backstop")
    p.add_argument("--cache", required=True)
    p.add_argument("--landmark-def", required=True)
    p.add_argument("--coord", type=_float_triple, required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--placed")
    p.add_argument("--frame")
    p.set_defaults(func=_cmd_backstop)

    p = sub.add_parser("emit")
    p.add_argument("--landmarks", required=True)
    p.add_argument("--workflow-orientation", required=True)
    p.add_argument("--spacing", type=_float_triple, required=True)
    p.add_argument("--source-artifacts", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=_cmd_emit)

    return parser


def _cmd_prepare(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.rendering import prepare_landmark_session, validate_segmentation_for_landmarking
    from microct_analysis.processing.session_cache import write_session_cache

    seg = _read_json(args.seg)
    session = prepare_landmark_session(seg)
    ok, reason = validate_segmentation_for_landmarking(session["labels"], session["assignments"])
    session["paths"] = {
        "labels": seg.get("labels"),
        "intensity": seg.get("intensity") or seg.get("filtered"),
    }
    write_session_cache(session, args.cache)
    return {
        "validation": {"ok": ok, "reason": reason},
        "bones": list(session["meshes"].keys()),
        "mesh_stats": {
            bone: {"render_verts": int(data["render"].n_points), "snap_verts": int(len(data["snap_vertices"]))}
            for bone, data in session["meshes"].items()
        },
    }


def _cmd_render_surface(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.rendering import render_surface_view
    from microct_analysis.processing.session_cache import load_session_cache

    session = load_session_cache(args.cache)
    mesh = _mesh(session, args.bone)["render"]
    camera = _read_json(args.camera)
    view_angle = float(camera.get("view_angle", 30.0))
    resolution = tuple(int(v) for v in camera.get("resolution", [1024, 1024]))
    render_surface_view(
        mesh,
        tuple(float(v) for v in camera["position"]),
        tuple(float(v) for v in camera["focal_point"]),
        tuple(float(v) for v in camera["view_up"]),
        resolution=resolution,
        view_angle=view_angle,
        annotations=camera.get("annotations"),
        output_path=args.out,
    )
    return {"image_path": args.out, "camera_params": {
        "position": [float(v) for v in camera["position"]],
        "focal_point": [float(v) for v in camera["focal_point"]],
        "view_up": [float(v) for v in camera["view_up"]],
        "view_angle": view_angle,
        "resolution": list(resolution),
    }}


def _cmd_render_slice(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.rendering import render_slice_view
    from microct_analysis.processing.session_cache import load_session_cache, load_volume

    session = load_session_cache(args.cache)
    volume_path = session["paths"][args.volume]
    volume = load_volume(volume_path)
    mask = None
    if args.mask_bone:
        labels = volume if args.volume == "labels" else load_volume(session["paths"]["labels"])
        mask = labels == session["assignments"][args.mask_bone]
    render_slice_view(volume, args.axis, args.index, args.window, mask=mask,
                      resolution=args.resolution, spacing=session["spacing"], output_path=args.out)
    slicing: list[Any] = [slice(None)] * 3
    slicing[args.axis] = args.index
    shape = list(np.asarray(volume[tuple(slicing)]).shape)
    return {"image_path": args.out, "axis": args.axis, "index": args.index, "window": list(args.window), "shape": shape}


def _cmd_snap_surface(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.session_cache import load_session_cache
    from microct_analysis.processing.snapping import snap_to_surface

    session = load_session_cache(args.cache)
    mesh_data = _mesh(session, args.bone)
    camera_payload = _read_json(args.camera)
    camera = camera_payload.get("camera_params", camera_payload)
    resolution = args.resolution or camera.get("resolution")
    if resolution is None:
        raise ValueError("resolution is required via --resolution or camera JSON camera_params.resolution")
    resolution = tuple(int(v) for v in resolution)
    result = snap_to_surface(args.pixel, camera, mesh_data["render"], mesh_data["snap_kdtree"],
                             mesh_data["snap_vertices"], resolution)
    return _rename_coordinate(result)


def _cmd_snap_slice(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.session_cache import load_session_cache, load_volume
    from microct_analysis.processing.snapping import snap_to_slice

    session = load_session_cache(args.cache)
    labels = load_volume(session["paths"]["labels"])
    mask = labels == session["assignments"][args.bone]
    result = snap_to_slice(args.slice, args.pixel, mask, session["spacing"], args.mode)
    return _rename_coordinate(result)


def _cmd_build_frame(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.femoral_frame import build_femoral_frame, serialize_derived_frame
    from microct_analysis.processing.session_cache import load_session_cache

    session = load_session_cache(args.cache)
    landmarks = _as_landmark_list(_read_json(args.landmarks))
    frame = build_femoral_frame(landmarks, _mesh(session, args.bone)["snap_vertices"])
    if frame is None:
        return {"frame": None}
    result = serialize_derived_frame(frame, _frame_source_landmarks(landmarks))
    result.update({
        "e_ML": frame.e_ML.tolist(),
        "e_AP": frame.e_AP.tolist(),
        "e_SI": frame.e_SI.tolist(),
        "confidence": frame.confidence,
        "evidence": frame.evidence,
    })
    return result


def _cmd_backstop(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.backstop import compute_backstop
    from microct_analysis.processing.session_cache import load_session_cache, load_volume

    session = load_session_cache(args.cache)
    landmark_def = _read_json(args.landmark_def)
    placed = _read_json(args.placed) if args.placed else None
    frame = _frame_from_json(_read_json(args.frame)) if args.frame else None
    domain = str(landmark_def.get("domain", ""))
    mesh_vertices = labels = intensity = None
    if domain == "femoral_3d_surface":
        mesh_vertices = _mesh(session, args.bone)["snap_vertices"]
    elif domain == "tibial_2d_slice":
        labels = load_volume(session["paths"]["labels"])
        intensity_path = session["paths"].get("intensity")
        intensity = load_volume(intensity_path) if intensity_path else None
    else:
        labels_path = session["paths"].get("labels")
        labels = load_volume(labels_path) if labels_path else None
    result = compute_backstop(landmark_def, args.coord, mesh_vertices=mesh_vertices, labels=labels,
                              intensity=intensity, spacing=session["spacing"], placed_landmarks=placed,
                              femoral_frame=frame)
    return asdict(result)


def _cmd_emit(args: argparse.Namespace) -> dict[str, Any]:
    from microct_analysis.processing.session_cache import load_session_cache
    from microct_analysis.stages.visual_landmarks import emit_positions

    session = load_session_cache(args.cache)
    return emit_positions(
        _as_landmark_list(_read_json(args.landmarks)),
        _read_json(args.workflow_orientation),
        spacing=args.spacing,
        source_artifacts=_read_json(args.source_artifacts),
        output_dir=args.out_dir,
        mesh_vertices=_mesh(session, args.bone)["snap_vertices"],
    )


def _read_json(path: str | None) -> Any:
    if path is None:
        return None
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return json.loads(text)


def _mesh(session: dict[str, Any], bone: str) -> dict[str, Any]:
    try:
        return session["meshes"][bone]
    except KeyError as exc:
        raise KeyError(f"bone not found in cache: {bone}") from exc


def _as_landmark_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "landmarks" in payload:
        return list(payload["landmarks"])
    return list(payload)


def _frame_source_landmarks(landmarks: list[dict[str, Any]]) -> list[str]:
    required = [
        "lateral_condylar_edge",
        "medial_condylar_edge",
        "intercondylar_groove_midpoint",
        "intercondylar_notch",
    ]
    available = {str(item.get("id")) for item in landmarks}
    return [landmark_id for landmark_id in required if landmark_id in available]


def _frame_from_json(payload: dict[str, Any]):
    from microct_analysis.processing.femoral_frame import FemoralFrame

    return FemoralFrame(
        e_ML=np.asarray(payload.get("e_ML", payload.get("ml_vector")), dtype=float),
        e_AP=np.asarray(payload.get("e_AP", payload.get("ap_vector")), dtype=float),
        e_SI=np.asarray(payload.get("e_SI", payload.get("si_vector")), dtype=float),
        confidence=str(payload.get("confidence", "low")),
        evidence=dict(payload.get("evidence", {})),
    )


def _rename_coordinate(result: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(result)
    if "coordinate" in renamed:
        renamed["coordinate_zyx"] = renamed.pop("coordinate")
    return renamed


def _int_pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return int(parts[0]), int(parts[1])


def _float_pair(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers")
    return float(parts[0]), float(parts[1])


def _float_triple(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
