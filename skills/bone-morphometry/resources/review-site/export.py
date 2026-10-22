"""Export the reviewed run into an explicit, portable web-review manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv

COLORS = [
    "#e3bb78",
    "#56b4e9",
    "#cc79a7",
    "#009e73",
    "#d55e00",
    "#f0e442",
    "#999999",
    "#486ab8",
    "#8057b8",
    "#b87848",
    "#579da3",
    "#a84c73",
    "#66bb66",
    "#e3bb78",
    "#d96eb4",
    "#49cbd4",
    "#4dd0e1",
    "#f5b45b",
]


def export(run, out, study_pdf=None):
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "prepared-3/scan.json").read_text())
    affine = np.array(meta["affine_xyz_to_world_mm"])
    center = affine[:3, :3] @ ((np.array(meta["shape_zyx"])[::-1] - 1) / 2) + affine[:3, 3]
    structures = json.loads((run / "review-07-separated/seeds.json").read_text())["structures"]
    labels = np.load(run / "review-07-separated/labels.npy", mmap_mode="r")
    counts = np.bincount(labels.ravel())
    for s in structures:
        i = s["label"]
        mesh = pv.read(run / f"review-07-separated/label-{i}.vtp")
        positions = np.asarray(mesh.points - center, dtype="<f4")
        indices = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype="<u4")
        path = out / f"label-{i}.bin"
        path.write_bytes(positions.tobytes() + indices.tobytes())
        s.update(
            file=path.name,
            vertices=len(positions),
            indices=indices.size,
            color=COLORS[i - 1],
            volume_mm3=float(counts[i] * abs(np.linalg.det(affine[:3, :3]))),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    measurements = json.loads((run / "final-measurements/measurements.json").read_text())
    for m in measurements.values():
        if "points_xyz" in m:
            p = np.array(m["points_xyz"])
            m["points_scene_mm"] = (p @ affine[:3, :3].T + affine[:3, 3] - center).tolist()
            assert abs(np.linalg.norm(np.diff(m["points_scene_mm"], axis=0)) - m["value"]) < 1e-6
    evidence = []
    # Explicit review directories only: never traverse the dataset or supplemental material.
    dirs = [
        "intake",
        "prepared-3",
        "scripts",
        "review-04-watershed",
        "review-05-bones",
        "review-06-corrected",
        "review-07-separated",
        "review-08-anatomical",
        "review-09-landmarks",
        "review-10-measurements",
        "final-measurements",
        "supplement-comparison",
        "placement-audit",
        "parameter-deck",
        "iioc",
        "tibial-trabecular",
    ]
    for directory in dirs:
        for p in sorted((run / directory).glob("*")):
            if p.suffix not in {".png", ".json", ".md", ".py", ".pdf", ".pptx"}:
                continue
            dest = out / "evidence" / directory / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            evidence.append(
                {
                    "path": str(dest.relative_to(out)),
                    "stage": directory,
                    "name": p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
            )
    if study_pdf is not None:
        dest = out / "evidence" / "study-protocol" / "main-study.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(study_pdf, dest)
        evidence.append(
            {
                "path": str(dest.relative_to(out)),
                "stage": "study-protocol",
                "name": dest.name,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            }
        )
    frames = json.loads((run / "review-08-anatomical/frames.json").read_text())
    manifest = dict(
        sample="OA6-1RK",
        grid_id=meta["grid_id"],
        affine=affine.tolist(),
        center_world_mm=center.tolist(),
        shape_zyx=meta["shape_zyx"],
        structures=structures,
        measurements=measurements,
        evidence=evidence,
        frames=frames,
        display="Surfaces sampled every 2 voxels; no geometry smoothing. Measurements use original coordinates and full-resolution label counts.",
    )
    manifest["parameter_overrides"] = {}
    for directory in ["iioc", "tibial-trabecular"]:
        additions = run / directory / "review-additions.json"
        if not additions.exists():
            continue
        extra = json.loads(additions.read_text())
        for structure in extra["structures"]:
            source = run / directory / structure["file"]
            destination = out / structure["file"]
            if source.exists() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        manifest["structures"].extend(extra["structures"])
        manifest["measurements"].update(extra["measurements"])
        manifest["parameter_overrides"].update(extra.get("parameter_overrides", {}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(structures)} structures and {len(evidence)} evidence files to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--study-pdf", type=Path)
    a = p.parse_args()
    export(a.run, a.output, a.study_pdf)
