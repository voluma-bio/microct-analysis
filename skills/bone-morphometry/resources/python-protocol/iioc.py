"""Explicit provisional IIOC segmentation from reviewable growth-plate traces.

Run from repository root: .venv/bin/python .../iioc.py data/python-protocol-run
No reference-table targets enter the segmentation or measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

SPACING = 0.0105
THRESHOLD = 2459.875


def boundary_surface(curves, shape):
    """Piecewise-linear U,V surface, constant continuation beyond traced extent."""
    levels = sorted(map(int, curves))
    rows = np.array([np.interp(np.arange(shape[2]), *np.array(curves[str(v)]).T) for v in levels])
    return np.array([np.interp(np.arange(shape[1]), levels, rows[:, u]) for u in range(shape[2])]).T


def envelope_from_tibia(labels, gp, iterations=2):
    # Seal the cut face before filling marrow cavities. A bone-only fill leaks
    # through vascular channels and through the growth-plate cut.
    result = np.zeros(labels.shape, bool)
    for v in range(labels.shape[1]):
        plane = labels[:, v, :] == 2
        # Remove thin segmentation bridges/spurs before constructing envelope.
        plane = ndi.binary_opening(plane, iterations=1)
        plane &= np.arange(labels.shape[0])[:, None] < gp[v][None, :]
        cc, _ = ndi.label(plane)
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        plane = sizes[cc] >= 50
        contact = plane & (np.arange(labels.shape[0])[:, None] >= gp[v][None, :] - 10)
        us = np.flatnonzero(contact.any(axis=0))
        if len(us) > 1:
            uu = np.arange(us[0], us[-1] + 1)
            plane[np.clip(np.ceil(gp[v, uu]).astype(int) - 1, 0, labels.shape[0] - 1), uu] = True
        closed = ndi.binary_closing(plane, iterations=iterations) | plane
        filled = ndi.binary_fill_holes(closed)
        result[:, v, :] = filled
    # Fill remaining enclosed 3D cavities without convexifying outer concavities.
    result = ndi.binary_fill_holes(ndi.binary_closing(result, iterations=1) | result)
    for u in range(labels.shape[2]):
        result[:, :, u] = ndi.binary_fill_holes(result[:, :, u])
    return result


def original_roi(aligned, frame, shape):
    b = np.array(frame["output_xyz_to_original_xyz_matrix"])
    offset = np.array(frame["output_xyz_to_original_xyz_offset"])
    # inverse map original XYZ -> aligned XYZ, with scipy ZYX reversal
    inverse = b.T
    return ndi.affine_transform(
        aligned.astype(np.uint8),
        inverse[::-1, ::-1],
        (-inverse @ offset)[::-1],
        output_shape=shape,
        order=0,
        prefilter=False,
    ).astype(bool)


def mesh_export(mask, frame, meta, center, path):
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1), 0.5, step_size=1)
    xyz = (vertices[:, ::-1] - 1) @ np.array(frame["output_xyz_to_original_xyz_matrix"]).T + frame[
        "output_xyz_to_original_xyz_offset"
    ]
    affine = np.array(meta["affine_xyz_to_world_mm"])
    positions = np.asarray(xyz @ affine[:3, :3].T + affine[:3, 3] - center, dtype="<f4")
    indices = np.asarray(faces, dtype="<u4")
    path.write_bytes(positions.tobytes() + indices.tobytes())
    return dict(
        file=path.name,
        vertices=len(positions),
        indices=indices.size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def run(root):
    out = root / "iioc"
    out.mkdir(exist_ok=True)
    annotations = json.loads((out / "annotations.json").read_text())
    h = np.load(root / "review-08-anatomical/tibia-hu.npy", mmap_mode="r")
    labels = np.load(root / "review-08-anatomical/tibia-labels.npy", mmap_mode="r")
    frame = json.loads((root / "review-08-anatomical/frames.json").read_text())["tibia"]
    meta = json.loads((root / "prepared-3/scan.json").read_text())
    gp = boundary_surface(annotations["growth_plate_curves"], h.shape)
    envelope = envelope_from_tibia(labels, gp)
    w = np.arange(h.shape[0])[:, None, None]
    roi = envelope & (w < gp[None])
    correction = annotations["articular_correction"]
    surface = boundary_surface(correction["curves"], h.shape)
    ulo, uhi = correction["u_range"]
    vlo, vhi = correction["v_range"]
    keep = np.ones(h.shape, bool)
    keep[:, vlo:vhi, ulo:uhi] = w >= surface[None, vlo:vhi, ulo:uhi]
    roi &= keep
    # Retain largest anatomical object; report removed specks instead of attaching
    # disconnected islands to the IIOC.
    components, n = ndi.label(roi)
    counts = np.bincount(components.ravel())
    counts[0] = 0
    removed = int(roi.sum() - counts.max())
    roi = components == counts.argmax()
    del components
    bone = roi & (h > THRESHOLD)
    np.save(out / "tissue-mask.npy", roi)
    np.save(out / "bone-mask.npy", bone)
    np.save(out / "growth-plate-proximal-face.npy", gp)
    raw = np.load(root / "intake/hu.npy", mmap_mode="r")
    filtered = np.load(root / "prepared-3/hu.npy", mmap_mode="r")
    original = original_roi(roi, frame, raw.shape)
    np.save(out / "tissue-mask-original.npy", original)
    tv = int(original.sum()) * SPACING**3
    bv = int(np.count_nonzero(raw[original] > THRESHOLD)) * SPACING**3
    measurements = {}

    def add(key, value, unit, definition, points=None, label=13, **extra):
        m = dict(
            value=float(value),
            unit=unit,
            definition=definition,
            label=label,
            status="Provisional: boundary/ROI review required",
            **extra,
        )
        if points is not None:
            p = np.asarray(points)
            m["points_aligned_uvw"] = p.tolist()
            m["points_xyz"] = (
                p @ np.array(frame["output_xyz_to_original_xyz_matrix"]).T + frame["output_xyz_to_original_xyz_offset"]
            ).tolist()
            affine = np.array(meta["affine_xyz_to_world_mm"])
            endpoints_mm = np.array(m["points_xyz"]) @ affine[:3, :3].T + affine[:3, 3]
            m["value"] = float(np.linalg.norm(endpoints_mm[1] - endpoints_mm[0]))
            m["distance_method"] = (
                "Direct Euclidean endpoint distance after tibial orientation correction; no slice counting"
            )
        m["bone"] = "tibia"
        measurements[key] = m

    add(
        "iioc_total_volume",
        tv,
        "mm^3",
        "Whole IIOC anatomical envelope above traced proximal growth-plate face, including marrow and cortical shell. Original-grid ROI voxel count.",
    )
    add(
        "iioc_bone_volume",
        bv,
        "mm^3",
        "Raw calibrated original-grid HU >2459.875 inside the same IIOC tissue ROI; includes shell, plate and trabeculae.",
        label=14,
    )
    add(
        "iioc_bv_tv", bv / tv, "1", "Whole-IIOC mineralized bone volume / tissue volume. Not isolated cancellous BV/TV."
    )
    valid = roi.any(axis=0)
    top = np.argmax(roi, axis=0).astype(float)
    local_height = np.where(valid, gp - top, np.nan)
    # Require a 3-voxel interior footprint to avoid tangential edge extrema.
    interior = ndi.binary_erosion(valid, iterations=3)
    search = interior.copy()
    loc = np.unravel_index(np.nanargmax(np.where(search, local_height, np.nan)), valid.shape)
    v, u = map(int, loc)
    add(
        "iioc_max_local_height",
        local_height[v, u] * SPACING,
        "mm",
        "Direct maximal tibial IIOC height after orientation correction: articular surface to proximal growth-plate face, parallel to corrected tibial shaft; searched all aligned slices.",
        [[u, v, top[v, u]], [u, v, gp[v, u]]],
    )
    # Global onset heights may occur at different positions: no artificial 3D
    # chord is constructed between them.
    proximal_top = float(top[interior].min())
    proximal_gp = float(gp[interior].min())
    add(
        "iioc_slice_count_height",
        (proximal_gp - proximal_top) * SPACING,
        "mm",
        "Difference of most-proximal surface and growth-plate-face W levels in tibia-aligned frame (paper slice-onset convention); endpoints need not share U,V.",
        surface_w=proximal_top,
        growth_plate_w=proximal_gp,
    )
    measurements["iioc_slice_count_height"]["status"] = "Retired by researcher instruction; historical diagnostic only"
    for side, midpoint in annotations["compartment_midpoint_u"].items():
        vi = v
        add(
            f"iioc_{side}_height",
            local_height[vi, midpoint] * SPACING,
            "mm",
            f"Height at provisional {side} plateau midpoint U{midpoint} on shared maximum-local-height coronal section V{vi}; shaft-direction ruler.",
            [[midpoint, vi, top[vi, midpoint]], [midpoint, vi, gp[vi, midpoint]]],
        )
    # Width endpoints at boundary footprint edges within each coronal section;
    # permit differing W (tilted chord as in user slide), but identical V.
    widths = []
    for vi in range(80, 310):
        near = roi[:, vi, :] & (w[:, 0, :] >= gp[vi][None, :] - 3)
        us = np.flatnonzero(near.any(axis=0))
        if len(us) < 2:
            continue
        left, right = int(us[0]), int(us[-1])
        a = [left, vi, gp[vi, left]]
        b = [right, vi, gp[vi, right]]
        widths.append((float(np.linalg.norm(np.subtract(a, b)) * SPACING), a, b))
    auto_width, auto_a, auto_b = max(widths, key=lambda x: x[0])
    a, b = annotations["width_reviewed_points_uvw"]
    width = float(np.linalg.norm(np.subtract(a, b)) * SPACING)
    add(
        "iioc_growth_plate_width",
        width,
        "mm",
        "Manually refined growth-plate junction chord at widest screened section V173; allows tilt. Maximal-width candidate: medial junction indistinct and exhaustive maximum not yet accepted.",
        [a, b],
    )
    add(
        "iioc_height_width",
        measurements["iioc_max_local_height"]["value"] / width,
        "1",
        "Direct orientation-corrected maximal tibial IIOC height divided by manually refined growth-plate width; provisional landmarks.",
    )
    plates = {}
    for side, patch in annotations["plate_patches"].items():
        mask = np.zeros_like(roi)
        thickness = []
        rays = []
        for vi in range(*patch["v"]):
            for ui in range(*patch["u"]):
                ww = np.flatnonzero(bone[:, vi, ui])
                if not len(ww):
                    continue
                start = int(ww[0])
                end = start
                while end < h.shape[0] and bone[end, vi, ui]:
                    end += 1
                # Keep the full first run; never cap thick columns to a desired
                # thickness. Flag continuity with deep trabeculae for review.
                mask[start:end, vi, ui] = True
                thickness.append((end - start) * SPACING)
                rays.append([[ui, vi, start - 0.5], [ui, vi, end - 0.5]])
        plates[side] = mask
        idx = int(np.argsort(thickness)[len(thickness) // 2])
        add(
            f"iioc_{side}_plate_thickness",
            np.median(thickness),
            "mm",
            f"Median first contiguous mineralized run along shaft direction in {side} 0.420×0.420 mm plateau patch. Directional plate-thickness candidate, not sphere-based 3D thickness; trabecular bridges can inflate it.",
            rays[idx],
            label=15 if side == "medial" else 16,
            patch=patch,
            mean_mm=float(np.mean(thickness)),
            p10_p90_mm=np.percentile(thickness, [10, 90]).tolist(),
            columns=len(thickness),
            columns_over_035mm=int(np.sum(np.array(thickness) > 0.35)),
        )
        np.save(out / f"{side}-plate-mask.npy", mask)
    # Sensitivity brackets are method perturbations, not confidence intervals.
    sensitivity = []
    for delta in [-2, 0, 2]:
        test = envelope_from_tibia(labels, gp + delta) & (w < (gp + delta)[None]) & keep
        cc, _ = ndi.label(test)
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        test = cc == sizes.argmax()
        mapped = original_roi(test, frame, raw.shape)
        tvn = int(mapped.sum()) * SPACING**3
        for threshold in [THRESHOLD, 2500.0]:
            bvn = int(np.count_nonzero(raw[mapped] > threshold)) * SPACING**3
            sensitivity.append(
                dict(boundary_shift_voxels=delta, threshold_hu=threshold, tv_mm3=tvn, bv_mm3=bvn, bv_tv=bvn / tvn)
            )
    stats = dict(
        status="Provisional, not accepted paper-equivalent measurements",
        automated_width_screen=dict(
            value_mm=auto_width, points_uvw=[auto_a, auto_b], status="Screen only; extrapolated edges not accepted"
        ),
        spacing_mm=SPACING,
        threshold_hu=THRESHOLD,
        annotations=annotations,
        measurements=measurements,
        sensitivity=sensitivity,
        removed_disconnected_voxels=removed,
        processing=dict(
            primary="Inverse-map anatomical ROI to original grid; threshold unfiltered calibrated HU",
            envelope="Existing tibia >2500 label, one-iteration opening and remove coronal components <50 voxels, growth-plate cut-face sealing, two-iteration coronal closing/hole fill, one-iteration 3D closing/hole fill and sagittal hole fill; clip at traced proximal face",
            aligned_bv_tv=float(bone.sum() / roi.sum()),
            original_filtered_bv_tv=float(np.count_nonzero(filtered[original] > THRESHOLD) / original.sum()),
        ),
        limitations=annotations["uncertainties"],
    )
    (out / "results.json").write_text(json.dumps(stats, indent=2) + "\n")
    for vs in ([80, 120, 160, 200], [220, 260, 290, 320]):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for vi, ax in zip(vs, axs.flat):
            ax.imshow(h[:, vi, :], cmap="gray", vmin=0, vmax=5000)
            ax.contour(roi[:, vi, :], levels=[0.5], colors=["lime"], linewidths=0.9)
            ax.plot(np.arange(h.shape[2]), gp[vi], color="orange", lw=0.8, label="Proximal growth-plate face")
            for side, color in [("medial", "magenta"), ("lateral", "cyan")]:
                if plates[side][:, vi, :].any():
                    ax.contour(plates[side][:, vi, :], levels=[0.5], colors=[color], linewidths=0.8)
            ax.set(
                xlim=(40, 390),
                ylim=(235, 55),
                title=f"V={vi}: green tissue envelope; orange growth-plate face",
                xlabel="U (lateral → medial)",
                ylabel="W (distal →)",
            )
            ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(out / f"qc-coronal-{vs[0]}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for wi, ax in zip([100, 140, 175], axs):
        ax.imshow(h[wi], cmap="gray", vmin=0, vmax=5000)
        ax.contour(roi[wi], levels=[0.5], colors="lime", linewidths=0.8)
        ax.set(title=f"Axial W={wi}", xlabel="U", ylabel="V")
    fig.tight_layout()
    fig.savefig(out / "qc-axial.png", dpi=130)
    plt.close(fig)
    ruler_items = [(k, m) for k, m in measurements.items() if "points_aligned_uvw" in m]
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    for (key, m), ax in zip(ruler_items, axs.flat):
        pts = np.array(m["points_aligned_uvw"])
        vi = int(round(pts[0, 1]))
        ax.imshow(h[:, vi, :], cmap="gray", vmin=0, vmax=5000)
        ax.contour(roi[:, vi, :], levels=[0.5], colors="lime", linewidths=0.7)
        ax.plot(pts[:, 0], pts[:, 2], "o-", color="cyan")
        ax.set(xlim=(20, 395), ylim=(230, 55), title=f"{key.replace('iioc_', '')}\n{m['value']:.4f} mm, V{vi}")
    fig.tight_layout()
    fig.savefig(out / "qc-rulers.png", dpi=130)
    plt.close(fig)
    # Sagittal independent check of interpolated boundary continuity.
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ui, ax in zip([140, 220, 290], axs):
        ax.imshow(h[:, :, ui], cmap="gray", vmin=0, vmax=5000)
        ax.contour(roi[:, :, ui], levels=[0.5], colors="lime", linewidths=0.8)
        ax.plot(np.arange(h.shape[1]), gp[:, ui], color="orange", lw=0.8)
        ax.set(xlim=(40, 339), ylim=(240, 50), title=f"Sagittal U={ui}", xlabel="V", ylabel="W")
    fig.tight_layout()
    fig.savefig(out / "qc-sagittal.png", dpi=130)
    plt.close(fig)
    web = root / "web-review"
    manifest = json.loads((web / "manifest.json").read_text())
    center = np.array(manifest["center_world_mm"])
    structures = []
    for label, name, mask, color in [
        (13, "IIOC tissue envelope (provisional)", roi, "#66bb66"),
        (14, "IIOC mineralized bone (provisional)", bone, "#e3bb78"),
        (15, "Medial plate patch (directional candidate)", plates["medial"], "#d96eb4"),
        (16, "Lateral plate patch (directional candidate)", plates["lateral"], "#49cbd4"),
    ]:
        s = dict(
            label=label,
            name=name,
            color=color,
            volume_mm3=float(mask.sum() * SPACING**3),
            definition="Aligned-grid display mask; primary whole-IIOC volume measured on original grid.",
        )
        s.update(mesh_export(mask, frame, meta, center, web / f"label-{label}.bin"))
        structures.append(s)
    affine = np.array(meta["affine_xyz_to_world_mm"])
    for m in measurements.values():
        if "points_xyz" in m:
            m["points_scene_mm"] = (np.array(m["points_xyz"]) @ affine[:3, :3].T + affine[:3, 3] - center).tolist()
    mapping = {
        7: "iioc_medial_plate_thickness",
        8: "iioc_lateral_plate_thickness",
        9: "iioc_total_volume",
        10: "iioc_bone_volume",
        11: "iioc_bv_tv",
        12: "iioc_max_local_height",
        13: "iioc_growth_plate_width",
        14: "iioc_height_width",
    }
    overrides = {
        str(i): dict(
            source_measurement=key,
            status="Directional candidate" if i in (7, 8) else "Provisional segmentation",
            labels=[measurements[key]["label"]],
            context="iioc",
            method=measurements[key]["definition"],
            unit=measurements[key]["unit"],
        )
        for i, key in mapping.items()
    }
    overrides["12"]["related_measurements"] = ["iioc_lateral_height", "iioc_medial_height"]
    # Preserve the old convention in machine-readable history only.
    measurements.pop("iioc_slice_count_height", None)
    (out / "review-additions.json").write_text(
        json.dumps(dict(structures=structures, measurements=measurements, parameter_overrides=overrides), indent=2)
        + "\n"
    )
    provenance = dict(
        command=f".venv/bin/python {Path(__file__)} {root}",
        input_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                out / "annotations.json",
                Path(__file__),
                root / "review-08-anatomical/frames.json",
                root / "prepared-3/scan.json",
            ]
        },
        inputs=["intake/hu.npy", "review-08-anatomical/tibia-hu.npy", "review-08-anatomical/tibia-labels.npy"],
        protocol="User-marked Pasted Graphic 13.tiff: whole IIOC excludes growth plate; retain marrow for TV. Researcher revision: use direct endpoint distances after tibial orientation correction, not slice counting.",
        judgment="Image-traced proximal growth-plate face; no supplementary numeric targets used. All results require boundary review. Plate first runs can include trabecular bridges and are not accepted anatomical plate thickness.",
        output_grid="Aligned isotropic 0.0105 mm masks plus inverse-mapped original-grid tissue mask; primary TV/BV on original grid.",
        qc=["qc-coronal-80.png", "qc-coronal-220.png", "qc-sagittal.png", "qc-axial.png", "qc-rulers.png"],
        edge_voxels=[int(np.take(roi, i, axis=j).sum()) for j in range(3) for i in [0, -1]],
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = "\n".join(
        f"| {key.replace('iioc_', '').replace('_', ' ')} | {m['value']:.5f} | {m['unit']} |"
        for key, m in measurements.items()
    )
    report = f"""# Tibial IIOC — OA6-1RK provisional segmentation

These measurements are for **tibial bone**. The [trabecular-only analysis](../tibial-trabecular/REPORT.md) is reported separately from these whole-IIOC volumes.

The tissue envelope follows an image-traced proximal growth-plate face and includes internal marrow spaces. It excludes the growth-plate band and metaphysis. These are review candidates, not accepted reference measurements.

| Measurement | Value | Unit |
|---|---:|---|
{rows}

BV/TV is a fraction; multiply by 100 for percent. This is whole-IIOC bone fraction including shell and plate, not cancellous-only BV/TV. TV and BV use the same inverse-mapped original-grid ROI; BV uses unfiltered calibrated HU >2459.875 (Scanco270 equivalent). Display bone uses resliced median-filtered HU, so display-mask volume differs from primary BV. Both processing alternatives are recorded in results.json.

## Definitions and review limits

Growth-plate traces are sampled every 20 aligned slices (210 µm); linear interpolation and constant edge continuation need review, particularly the medial bridges and anterior/posterior margins. Moving the traced face ±2 voxels is a sensitivity check, not a confidence interval or full estimate of manual-placement error.

Per researcher instruction, primary heights are direct Euclidean distances between saved endpoints after tibial shaft-based orientation correction. Slice-count height is retired and excluded from the height/width ratio. The maximum local height is the greatest shaft-direction endpoint distance in the eroded footprint. Compartment heights use provisional midpoint U140 (lateral) and U290 (medial) on the shared maximum-local-height coronal slice. The width is a manually refined tilted chord at the widest screened section V173; it is a maximum candidate, not an exhaustively reviewed maximum. These definitions are kept distinct.

**Plate thickness is not yet an accepted anatomical measurement.** The displayed patches are the first contiguous mineralized run in 0.420×0.420 mm sampling windows. Trabecular bridges visibly extend some runs. The median and distribution are directional diagnostics; a reviewed plate-to-trabecula boundary is still required. Do not treat these numbers as the paper's surface-to-first-trabecular-appearance thickness.

Independent final review found no remaining coordinate/arithmetic blocker for provisional reporting, but a small separate-in-section region near U370,V200 still needs anatomical ownership review before accepting TV/BV.

The first run left marrow spaces out of TV because of openings in the shell (round1-results.json and round1-sagittal.png). Revised envelopes seal the growth-plate cut before filling holes. Small coronal fragments (<50 voxels) from the prior tibia segmentation are excluded; they otherwise create false proximal height extrema. An additional image-traced articular-face correction removes a retained label2 fragment against the femur (U120:210,V150:181); see annotations.json. Anatomical edge choice and morphology operations remain sources of uncertainty.

## Review evidence

- [Coronal contours](qc-coronal-80.png) and [anterior contours](qc-coronal-220.png)
- [Sagittal contours](qc-sagittal.png) and [axial contours](qc-axial.png)
- [Measurement rulers](qc-rulers.png)
- [Exact traces](annotations.json), [results and sensitivity](results.json), [provenance](provenance.json)

Green is the tissue boundary, orange is the growth-plate proximal face; cyan/magenta are lateral/medial directional plate candidates. Existing final-measurements/measurements.json is preserved. NumPy masks are retained in this local run directory (Gitignored).
"""
    (out / "REPORT.md").write_text(report)

    print(json.dumps({k: round(m["value"], 5) for k, m in measurements.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    run(parser.parse_args().run)
