"""Tibial IIOC trabecular compartment: explicit contours and 3D sphere maps."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iioc import boundary_surface, mesh_export, SPACING, THRESHOLD


def local_thickness(phase, step=0.5):
    """Largest contained sphere covering each voxel, quantized radii in voxels.

    Distances use voxel centers. False padding makes ROI clipping explicit.
    Diameters have <2*step voxel radius-bin underestimation (plus discretization).
    This is NOT the mean of twice the distance transform.
    """
    phase = np.pad(np.asarray(phase, dtype=bool), 1)
    if not phase.any():
        raise ValueError("Empty phase")
    dt = ndi.distance_transform_edt(phase)
    radii = np.arange(1, np.floor(dt.max() / step) * step + step / 2, step)[::-1]
    result = np.zeros(phase.shape, np.float32)
    for r in radii:
        centers = dt >= r
        if not centers.any():
            continue
        covered = ndi.distance_transform_edt(~centers) < r
        result[(result == 0) & phase & covered] = 2 * r
    assert np.all(result[phase] > 0)
    return result[1:-1, 1:-1, 1:-1]


def compartment(parent, gp, annotations, upper_shift=0):
    curves = annotations["upper_inner_boundary_curves"]
    upper = boundary_surface(curves, parent.shape) + upper_shift
    vs = sorted(map(int, curves))
    left = np.interp(np.arange(parent.shape[1]), vs, [curves[str(v)][0][0] for v in vs])
    right = np.interp(np.arange(parent.shape[1]), vs, [curves[str(v)][-1][0] for v in vs])
    w, v, u = np.ogrid[: parent.shape[0], : parent.shape[1], : parent.shape[2]]
    inner = ndi.distance_transform_edt(parent) > annotations["outer_shell_clearance_voxels"]
    roi = inner & (w >= upper[None]) & (w < gp[None] - annotations["growth_face_clearance_voxels"])
    roi &= (u >= left[None, :, None]) & (u <= right[None, :, None])
    roi &= (v >= annotations["v_limits"][0]) & (v <= annotations["v_limits"][1])
    return roi, upper


def run(root):
    out = root / "tibial-trabecular"
    out.mkdir(exist_ok=True)
    consumed_paths = [
        root / "iioc/tissue-mask.npy",
        root / "iioc/growth-plate-proximal-face.npy",
        root / "review-08-anatomical/tibia-hu.npy",
        out / "annotations.json",
    ]
    consumed_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in consumed_paths}
    a = json.loads((out / "annotations.json").read_text())
    parent = np.load(root / "iioc/tissue-mask.npy")
    gp = np.load(root / "iioc/growth-plate-proximal-face.npy")
    hu = np.load(root / "review-08-anatomical/tibia-hu.npy", mmap_mode="r")
    roi, upper = compartment(parent, gp, a)
    bone = roi & (hu > THRESHOLD)
    marrow = roi & ~bone
    assert not np.any(roi & ~parent)
    assert np.array_equal(bone | marrow, roi)
    np.save(out / "tissue-mask.npy", roi)
    np.save(out / "bone-mask.npy", bone)
    np.save(out / "marrow-mask.npy", marrow)
    np.save(out / "upper-inner-boundary.npy", upper)
    # Write geometric QC before running the slower sphere propagation.
    for vs in [[100, 130, 160, 190], [220, 250, 280, 300]]:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for vi, ax in zip(vs, axs.flat):
            ax.imshow(hu[:, vi, :], cmap="gray", vmin=0, vmax=5000)
            ax.contour(parent[:, vi, :], levels=[0.5], colors="lime", linewidths=0.6)
            ax.contour(roi[:, vi, :], levels=[0.5], colors="cyan", linewidths=1)
            ax.contour(bone[:, vi, :], levels=[0.5], colors="orange", linewidths=0.5)
            ax.set(
                xlim=(40, 380),
                ylim=(220, 65),
                title=f"Tibial IIOC trabecular ROI | V{vi}",
                xlabel="U (lateral → medial)",
                ylabel="W (distal →)",
            )
        fig.tight_layout()
        fig.savefig(out / f"qc-coronal-{vs[0]}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ui, ax in zip([140, 220, 290], axs):
        ax.imshow(hu[:, :, ui], cmap="gray", vmin=0, vmax=5000)
        ax.contour(roi[:, :, ui], levels=[0.5], colors="cyan", linewidths=1)
        ax.set(xlim=(70, 320), ylim=(230, 60), title=f"Tibial trabecular ROI | U{ui}", xlabel="V", ylabel="W")
    fig.tight_layout()
    fig.savefig(out / "qc-sagittal.png", dpi=130)
    plt.close(fig)
    box = ndi.find_objects(roi.astype(np.uint8))[0]
    th = local_thickness(bone[box]) * SPACING
    print("Bone sphere map complete", flush=True)
    sp = local_thickness(marrow[box]) * SPACING
    print("Marrow sphere map complete", flush=True)
    np.save(out / "thickness-map-crop-mm.npy", th)
    np.save(out / "separation-map-crop-mm.npy", sp)
    tv = float(roi.sum() * SPACING**3)
    bv = float(bone.sum() * SPACING**3)
    fraction = bv / tv
    tbth = float(th[bone[box]].mean())
    tbsp = float(sp[marrow[box]].mean())
    metrics = {"TV": tv, "BV": bv, "BV/TV": fraction, "Tb.Th": tbth, "Tb.N": fraction / tbth, "Tb.Sp": tbsp}
    units = {"TV": "mm^3", "BV": "mm^3", "BV/TV": "1", "Tb.Th": "mm", "Tb.N": "1/mm", "Tb.Sp": "mm"}
    # One-voxel exclusion changes the sampling population, not the sphere maps.
    interior = ndi.binary_erosion(roi)[box]
    sensitivity = []
    for shift in [-2, 0, 2]:
        r, _ = compartment(parent, gp, a, upper_shift=shift)
        for threshold in [THRESHOLD, 2500.0]:
            b = r & (hu > threshold)
            sensitivity.append(
                dict(
                    upper_boundary_shift_voxels=shift,
                    threshold_hu=threshold,
                    TV_mm3=float(r.sum() * SPACING**3),
                    BV_mm3=float(b.sum() * SPACING**3),
                    BV_TV=float(b.sum() / r.sum()),
                )
            )
    summary = dict(
        bone="tibia",
        compartment=a["compartment"],
        status="Provisional anatomical contours; not validated Scanco-equivalent output",
        metrics=metrics,
        units=units,
        voxel_mm=SPACING,
        threshold_hu=THRESHOLD,
        processing="All six primary metrics use the same tibia-aligned, median-filtered then linearly resliced HU grid. BV/TV is a fraction.",
        methods={
            "BV/TV": "Mineralized bone volume divided by total trabecular compartment tissue volume",
            "TV": "ROI voxel count × voxel volume, includes trabeculae and marrow",
            "BV": "Thresholded mineralized voxels within exactly the same ROI",
            "Tb.Th": "Mean 3D largest-contained-sphere diameter over bone voxels; radii quantized at 0.5 voxels",
            "Tb.Sp": "Mean 3D largest-contained-sphere diameter over ROI marrow voxels; exterior excluded",
            "Tb.N": "Model-derived (BV/TV)/Tb.Th with Tb.Th in mm, not direct 3D Scanco Tb.N",
        },
        crop_slices_zyx=[[s.start, s.stop] for s in box],
        sensitivity=sensitivity,
        boundary_excluded_summary=dict(
            retained_TV_mm3=float(interior.sum() * SPACING**3),
            Tb_Th_mm=float(th[bone[box] & interior].mean()),
            Tb_Sp_mm=float(sp[marrow[box] & interior].mean()),
            meaning="Same maps sampled after excluding one boundary-voxel layer; diagnostic, not an alternative full-compartment result",
        ),
        limitations=[
            "Manual sub-plate boundary and endocortical limits require review, especially dense medial transition.",
            "ROI-clipped spheres can underestimate structures intersecting ROI boundaries.",
            "0.5-voxel radius quantization gives <10.5µm diameter bin error, additional to voxelization and segmentation errors.",
            "Tb.N is derived under a model assumption; not direct 3D trabecular number.",
            a["deviation_from_paper"],
        ],
    )
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    vi = 190 - box[1].start
    for arr, mask, title, ax in [(th, bone[box], "Tb.Th", axs[0]), (sp, marrow[box], "Tb.Sp", axs[1])]:
        img = ax.imshow(np.where(mask[:, vi, :], arr[:, vi, :] * 1000, np.nan), cmap="viridis", vmin=0)
        fig.colorbar(img, ax=ax, label="µm")
        ax.set_title(f"Tibial trabecular {title} | V190")
    fig.tight_layout()
    fig.savefig(out / "qc-sphere-maps.png", dpi=140)
    plt.close(fig)
    frame = json.loads((root / "review-08-anatomical/frames.json").read_text())["tibia"]
    meta = json.loads((root / "prepared-3/scan.json").read_text())
    manifest = json.loads((root / "web-review/manifest.json").read_text())
    center = np.array(manifest["center_world_mm"])
    structures = []
    for label, name, mask, color in [
        (17, "Tibial IIOC trabecular tissue ROI", roi, "#4dd0e1"),
        (18, "Tibial IIOC trabecular bone", bone, "#f5b45b"),
    ]:
        s = dict(
            label=label,
            name=name,
            color=color,
            volume_mm3=float(mask.sum() * SPACING**3),
            definition="Tibial IIOC trabecular compartment; cortical shell, subchondral plate and growth plate excluded by provisional anatomical contours.",
        )
        s.update(mesh_export(mask, frame, meta, center, out / f"label-{label}.bin"))
        structures.append(s)
    measurements = {}
    for metric, val in metrics.items():
        key = {"TV": "tv", "BV": "bv", "BV/TV": "bv_tv", "Tb.Th": "tb_th", "Tb.N": "tb_n", "Tb.Sp": "tb_sp"}[metric]
        measurements["tibial_trab_" + key] = dict(
            title="Tibial IIOC trabecular " + metric + (" (model-derived)" if metric == "Tb.N" else ""),
            value=val,
            unit=units[metric],
            label=18 if metric in ["BV", "Tb.Th", "Tb.N"] else 17,
            status="Provisional trabecular ROI",
            definition=summary["methods"][metric] + "; all six metrics use the same aligned HU segmentation.",
            evidence_stage="tibial-trabecular",
        )
    (out / "review-additions.json").write_text(
        json.dumps(dict(structures=structures, measurements=measurements), indent=2) + "\n"
    )
    sources = {
        "local_thickness": "https://porespy.org/autoapi/porespy/filters/local_thickness_dt.html",
        "bonej": "https://imagej.net/plugins/bonej",
        "Tb_N_model": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3672237/",
    }
    provenance = dict(
        consumed_array_hashes=consumed_hashes,
        command=f".venv/bin/python {Path(__file__)} {root}",
        sources=sources,
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [out / "annotations.json", Path(__file__), root / "iioc/provenance.json"]
        },
        judgment="Trace an anatomical trabecular compartment rather than subtracting first-run plate candidates or selecting marrow pores. No reference-table values enter ROI or metric selection.",
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = "\n".join(
        f"| {k}"
        + (" (model-derived)" if k == "Tb.N" else "")
        + f" | {val * 100 if k == 'BV/TV' else val:.5f} | {'%' if k == 'BV/TV' else units[k]} |"
        for k, val in metrics.items()
    )
    heights = json.loads((root / "iioc/review-additions.json").read_text())["measurements"]
    height_rows = "\n".join(
        f"| {label} | {heights[key]['value']:.5f} | {heights[key]['unit']} |"
        for key, label in [
            ("iioc_max_local_height", "Direct maximal tibial IIOC height"),
            ("iioc_medial_height", "Direct medial compartment height"),
            ("iioc_lateral_height", "Direct lateral compartment height"),
            ("iioc_growth_plate_width", "Growth-plate width candidate"),
            ("iioc_height_width", "Direct maximal height / width"),
        ]
    )
    report = f"""# Tibial IIOC trabecular analysis — OA6-1RK

These measurements concern **tibial bone**, specifically the trabecular portion of the tibial secondary ossification center (IIOC). The anatomical ROI includes trabeculae and intervening marrow, with contours intended to exclude cortical shell, subchondral plate and growth plate. Exclusion remains provisional, especially at the dense medial plate-to-trabecula transition.

| Tibial trabecular metric | Value | Unit |
|---|---:|---|
{rows}

All six metrics use the same 10.5µm tibia-aligned, median-filtered and linearly resliced HU grid; bone threshold >2459.875HU. These values replace whole-IIOC values for the requested **trabecular** analysis. The earlier whole-IIOC TV/BV/BVTV include shell and plate and remain separate historical results.

Tb.Th and Tb.Sp are phase-volume-weighted averages of largest-inscribed-sphere diameter maps. They are not averages of twice the ordinary distance transform. Separation is computed only in ROI marrow, excluding exterior background. Radii are quantized at 0.5 voxels. ROI truncation and resolution affect the results; boundary-excluded summaries and contour/threshold sensitivity are in [results](results.json). Excluding one sampling layer changes Tb.Th from {tbth * 1000:.2f} to {summary["boundary_excluded_summary"]["Tb_Th_mm"] * 1000:.2f} µm and Tb.Sp from {tbsp * 1000:.2f} to {summary["boundary_excluded_summary"]["Tb_Sp_mm"] * 1000:.2f} µm. This changes the sampled population; it does not quantify or correct clipped-sphere bias.

**Tb.N is model-derived as (BV/TV)/Tb.Th (mm⁻¹), not a direct 3D trabecular count.** This implementation is not validated as Scanco-equivalent. The strictly sub-plate IIOC ROI requested here is narrower than the paper's description including all solid and trabecular noncortical tissue through portions of the growth-plate region.

## Direct tibial heights after orientation correction

| Direct tibial measurement | Value | Unit |
|---|---:|---|
{height_rows}

The revised primary heights use saved 3D ruler endpoints in the shaft-aligned tibial frame. Slice-count height is retired from primary reporting and normalization. The maximal local height, medial height and lateral height are direct distances; rotation changes their anatomical placement but preserves Euclidean distance. Their endpoint definitions and provisional growth-plate face remain reviewable in the IIOC evidence.

## Review

- [Posterior/central coronal boundaries](qc-coronal-100.png), [anterior boundaries](qc-coronal-220.png)
- [Sagittal boundaries](qc-sagittal.png), [sphere maps](qc-sphere-maps.png)
- [Annotations](annotations.json), [all results](results.json), [provenance](provenance.json)

Cyan = trabecular tissue ROI; orange = mineralized trabeculae; green = parent whole-IIOC boundary. Dense medial plate-to-trabecula transition is uncertain and requires anatomical review. Full masks and diameter maps are preserved locally in this run directory.

Methods: [sphere propagation]({sources["local_thickness"]}), [BoneJ phase-map summaries]({sources["bonej"]}), [ASBMR model-derived trabecular number]({sources["Tb_N_model"]}).
"""
    (out / "REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    run(p.parse_args().run)
