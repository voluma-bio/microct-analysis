"""Explicit Python implementation of the supplied OA paper's imaging techniques.

Run with --help. Manual seeds, anatomical axes, ROIs, and landmarks are inputs,
never inferred from expected measurements. No supplemental tables are read.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk
from skimage.segmentation import watershed

from scan import grid_identity, load_series, read_scan
from visuals import render_review


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_array(path, meta):
    """External arrays require an explicit sidecar binding their bytes and coordinate grid."""
    path = Path(path)
    binding = json.loads(path.with_suffix(".json").read_text())
    if binding["grid_id"] != meta["grid_id"] or binding["sha256"] != file_hash(path):
        raise ValueError(f"Array content or grid identity mismatch: {path}")
    array = np.load(path, mmap_mode="r")
    if list(array.shape) != meta["shape_zyx"]:
        raise ValueError(f"Array shape mismatch: {path}")
    return array, dict(path=str(path.resolve()), **binding)


def save_bound_array(path, array, meta):
    np.save(path, array)
    write_json(Path(path).with_suffix(".json"), dict(grid_id=meta["grid_id"], sha256=file_hash(path)))


def record(output, action, parameters, inputs, status):
    output = Path(output)
    payload = dict(action=action, time_utc=datetime.now(timezone.utc).isoformat(),
                   command=sys.argv, parameters=parameters, inputs=inputs, status=status,
                   versions={p: version(p) for p in ["numpy", "scipy", "scikit-image", "pydicom", "SimpleITK", "pyvista"]},
                   code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py")},
                   artifacts={p.name: file_hash(p) for p in output.iterdir()
                              if p.is_file() and p.name != "provenance.json"})
    write_json(output / "provenance.json", payload)


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output must be empty to preserve earlier review rounds: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def connected_preview(hu, threshold, min_voxels):
    labels, _ = ndi.label(hu > threshold)
    counts = np.bincount(labels.ravel())
    ids = np.argsort(counts[1:])[::-1] + 1
    ids = ids[counts[ids] >= min_voxels]
    remap = np.zeros(len(counts), dtype=np.uint16)
    if len(ids) > 65535:
        raise ValueError("Too many components")
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[labels], [dict(label=i+1, voxels=int(counts[v])) for i, v in enumerate(ids)]


def seed_markers(filtered, seed_spec, lower=3000, upper=5000):
    """Magic-wand equivalent: connected in-range regions containing recorded seed voxels."""
    components, _ = ndi.label((filtered >= lower) & (filtered <= upper))
    mapping = np.zeros(int(components.max()) + 1, dtype=np.int32)
    ids = set()
    for region in seed_spec:
        label = int(region["label"])
        if label <= 0 or label in ids:
            raise ValueError("Seed labels must be unique positive integers")
        ids.add(label)
        if not region["points_zyx"]:
            raise ValueError("Each structure needs at least one seed")
        for point in region["points_zyx"]:
            p = np.asarray(point)
            if p.shape != (3,) or not np.equal(p, np.floor(p)).all() or (p < 0).any() or (p >= filtered.shape).any():
                raise ValueError(f"Seed outside image or not an integer ZYX index: {point}")
            component = int(components[tuple(p.astype(int))])
            if component == 0:
                raise ValueError(f"Seed {point} is outside the paper's marker intensity interval")
            if mapping[component] not in (0, label):
                raise ValueError("Two structures share a connected seed region; refine the seed mask")
            mapping[component] = label
    return mapping[components]


def physical_distance(points, affine):
    points = np.asarray(points, dtype=float)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("Distance needs two finite XYZ voxel-center coordinates")
    xyz = points @ affine[:3, :3].T + affine[:3, 3]
    return float(np.linalg.norm(xyz[1] - xyz[0]))


def measure(spec, meta, labels=None, hu=None):
    """Geometry uses recorded points; ROI BV/TV uses an explicit tissue denominator."""
    if spec.get("grid_id") != meta["grid_id"]:
        raise ValueError("Measurement annotations do not match scan/grid identity")
    affine = np.array(meta["affine_xyz_to_world_mm"])
    shape_xyz = np.array(meta["shape_zyx"])[::-1]
    measurements = {}
    for name, item in spec.get("distances", {}).items():
        points = np.array(item["points_xyz"], dtype=float)
        if np.any(points < 0) or np.any(points > shape_xyz - 1):
            raise ValueError(f"{name}: landmark outside image")
        measurements[name] = dict(value=physical_distance(points, affine), unit="mm",
                                  points_xyz=points.tolist(), definition=item["definition"])
    for name, item in spec.get("slice_heights", {}).items():
        for field in ["axis_xyz", "start_index", "end_index"]:
            if not isinstance(item[field], int) or isinstance(item[field], bool):
                raise ValueError(f"{field} must be an integer, not a rounded coordinate")
        axis = int(item["axis_xyz"])
        start, end = int(item["start_index"]), int(item["end_index"])
        if axis not in (0, 1, 2) or not 0 <= start < shape_xyz[axis] or not 0 <= end < shape_xyz[axis]:
            raise ValueError("Invalid slice-height indices")
        if not item.get("orientation_reviewed"):
            raise ValueError("Slice heights require reviewed anatomical orientation")
        # Index-center separation: N intervals, not an inclusive count of N+1 slices.
        intervals = abs(end-start)
        measurements[name] = dict(value=float(intervals * np.linalg.norm(affine[:3, axis])), unit="mm",
                                  intervals=intervals, definition=item["definition"],
                                  convention="slice-center separation; end-start")
    for name, item in spec.get("ratios", {}).items():
        numerator, denominator = measurements[item["numerator"]], measurements[item["denominator"]]
        if numerator["unit"] != denominator["unit"] or denominator["value"] <= 0:
            raise ValueError("Ratio needs compatible units and positive denominator")
        measurements[name] = dict(value=numerator["value"]/denominator["value"], unit="1", **item)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    if labels is not None:
        if (list(labels.shape) != meta["shape_zyx"]
                or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0):
            raise ValueError("Labels must match the measurement grid")
        for name, label in spec.get("label_volumes", {}).items():
            count = int(np.count_nonzero(labels == label))
            if count == 0:
                raise ValueError(f"Missing label for {name}")
            measurements[name] = dict(value=count * voxel_volume, unit="mm^3", label=label)
    elif spec.get("label_volumes"):
        raise ValueError("Label volumes requested without labels")
    for name, item in spec.get("regions", {}).items():
        if hu is None:
            raise ValueError("ROI measurements require HU volume")
        roi, binding = bound_array(item["mask_npy"], meta)
        if roi.dtype != bool or roi.shape != hu.shape or not roi.any():
            raise ValueError("ROI must be a nonempty boolean array on the same grid")
        threshold = float(item["threshold_hu"])
        if not np.isfinite(threshold):
            raise ValueError("ROI threshold must be finite")
        bone = np.count_nonzero(hu[roi] > threshold)
        tissue = np.count_nonzero(roi)
        measurements[name] = dict(bv_mm3=bone * voxel_volume, tv_mm3=tissue * voxel_volume,
                                  bv_tv_percent=100 * bone / tissue, definition=item["definition"],
                                  threshold_hu=item["threshold_hu"], mask=binding)
    return measurements


def reslice(hu, meta, axes):
    """Resample onto anatomical axes expressed in world coordinates, without cropping."""
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6) or np.linalg.det(axes) < 0:
        raise ValueError("Axes must be a right-handed orthonormal matrix (columns = X,Y,Z)")
    affine = np.array(meta["affine_xyz_to_world_mm"])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    image = sitk.GetImageFromArray(np.asarray(hu))
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(affine[:3, 3].tolist())
    image.SetDirection((affine[:3, :3] / spacing).ravel().tolist())
    # Include voxel support, not just center bounds.
    corners = np.array(list(itertools.product(*[(-0.5, n-0.5) for n in hu.shape[::-1]])))
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    coordinates = world @ axes
    step = float(min(spacing))
    origin = axes @ (coordinates.min(axis=0) + step/2)
    size = np.ceil(np.ptp(coordinates, axis=0)/step).astype(int)
    result = sitk.Resample(image, size.tolist(), sitk.Transform(), sitk.sitkLinear,
                          origin.tolist(), [step]*3, axes.ravel().tolist(), -1000, sitk.sitkFloat32)
    new_affine = np.eye(4)
    new_affine[:3, :3] = axes * step
    new_affine[:3, 3] = origin
    out = sitk.GetArrayFromImage(result)
    updated = dict(meta, shape_zyx=list(out.shape), spacing_xyz_mm=[step]*3,
                   affine_xyz_to_world_mm=new_affine.tolist(),
                   reslice=dict(axes_world=axes.tolist(), interpolation="linear", outside_hu=-1000))
    updated["grid_id"] = grid_identity(updated)
    return out, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    intake = sub.add_parser("intake")
    intake.add_argument("source", type=Path)
    intake.add_argument("output", type=Path)
    preview = sub.add_parser("preview", help="Raw threshold components for visual inspection, not anatomical labels")
    prep = sub.add_parser("prepare", help="Apply an explicitly selected 3D median size")
    seg = sub.add_parser("segment")
    rotate = sub.add_parser("reslice")
    metrics = sub.add_parser("measure")
    for cmd in [preview, prep, seg, rotate, metrics]:
        cmd.add_argument("run", type=Path)
        cmd.add_argument("output", type=Path)
    preview.add_argument("--threshold-hu", type=float, default=2500)
    preview.add_argument("--min-voxels", type=int, default=1000)
    preview.add_argument("--display-labels", type=int, nargs="+", help="Only affects 3D display, not saved labels or slice overlays")
    prep.add_argument("--median-size", type=int, required=True)
    seg.add_argument("--seeds", type=Path, required=True)
    seg.add_argument("--markers", type=Path, help="Optional reviewed integer marker array with .json grid/hash sidecar")
    rotate.add_argument("--axes", type=Path, required=True, help="JSON 3x3 anatomical axes in DICOM world coordinates")
    metrics.add_argument("--spec", type=Path, required=True)
    metrics.add_argument("--labels", type=Path)
    args = parser.parse_args()
    out = new_output(args.output)
    if args.action == "intake":
        meta = load_series(args.source, out)
        record(out, "intake", meta["calibration"], [str(args.source)], meta["status"])
        print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=2))
        return
    hu, meta = read_scan(args.run)
    inputs = [str(args.run.resolve())]
    params = {}
    status = "Preliminary: anatomy and protocol choices require visual review"
    if args.action == "preview":
        params = dict(threshold_hu=args.threshold_hu, min_voxels_display=args.min_voxels,
                      display_labels=args.display_labels,
                      method="6-connected components; no filtering or anatomical assignment")
        labels, components = connected_preview(hu, args.threshold_hu, args.min_voxels)
        save_bound_array(out / "labels.npy", labels, meta)
        write_json(out / "components.json", components)
        render_review(hu, labels, meta, out, display_labels=args.display_labels)
    elif args.action == "prepare":
        if args.median_size < 1 or args.median_size % 2 == 0:
            raise ValueError("Median size must be a positive odd integer")
        filtered = ndi.median_filter(hu, size=args.median_size, mode="nearest")
        np.save(out / "hu.npy", filtered)
        params = dict(median_size_zyx=[args.median_size]*3, boundary_mode="nearest",
                      note="Paper specifies 3D median filtering but not kernel size")
        write_json(out / "scan.json", dict(meta, preparation=params))
    elif args.action == "segment":
        if "preparation" not in meta:
            raise ValueError("Run prepare first: the paper's watershed input is median filtered")
        seed_document = json.loads(args.seeds.read_text())
        if seed_document.get("grid_id") != meta["grid_id"]:
            raise ValueError("Seed annotations do not match scan/grid identity")
        seeds = seed_document["structures"]
        if args.markers:
            markers, binding = bound_array(args.markers, meta)
            if not np.issubdtype(markers.dtype, np.integer) or markers.min() < 0:
                raise ValueError("Markers must be nonnegative integer labels")
            selected = markers > 0
            if not selected.any() or np.any((hu[selected] < 3000) | (hu[selected] > 5000)):
                raise ValueError("Manual markers must lie within 3000–5000 HU")
            if set(np.unique(markers[selected])) != {int(s["label"]) for s in seeds}:
                raise ValueError("Marker labels must match named seed structures")
            inputs.append(binding)
        else:
            markers = seed_markers(hu, seeds)
        mask = hu > 2500
        # Gradient magnitude is an explicit implementation choice; Amira's
        # watershed elevation and connectivity are not specified in the PDF.
        image = sitk.GetImageFromArray(np.asarray(hu))
        image.SetSpacing(meta["spacing_xyz_mm"])
        gradient = sitk.GetArrayFromImage(sitk.GradientMagnitude(image))
        labels = watershed(gradient, markers=markers, mask=mask, connectivity=1)
        save_bound_array(out / "labels.npy", labels, meta)
        save_bound_array(out / "markers.npy", markers, meta)
        write_json(out / "seeds.json", seed_document)
        write_json(out / "scan.json", meta)
        unassigned = int(np.count_nonzero(mask & (labels == 0)))
        params = dict(mask_hu=2500, marker_hu=[3000, 5000], elevation="gradient magnitude",
                      connectivity=6, unassigned_mask_voxels=unassigned, seeds=seeds)
        render_review(hu, labels, meta, out, {str(s["label"]): s["name"] for s in seeds})
        inputs.append(str(args.seeds.resolve()))
    elif args.action == "reslice":
        axes = json.loads(args.axes.read_text())
        rotated, updated = reslice(hu, meta, axes)
        np.save(out / "hu.npy", rotated)
        write_json(out / "scan.json", updated)
        params = updated["reslice"]
        inputs.append(str(args.axes.resolve()))
    elif args.action == "measure":
        spec = json.loads(args.spec.read_text())
        labels = None
        if args.labels:
            labels, binding = bound_array(args.labels, meta)
            inputs.append(binding)
        result = measure(spec, meta, labels, hu)
        write_json(out / "measurements.json", result)
        write_json(out / "measurement-spec.json", spec)
        inputs.append(str(args.spec.resolve()))
        if args.labels:
            inputs.append(str(args.labels.resolve()))
        params = spec
        status = "Computed from supplied annotations; scientific acceptance requires annotation/QC review"
    record(out, args.action, params, inputs, status)
    print(f"{args.action}: {out}\n{status}")


if __name__ == "__main__":
    main()
