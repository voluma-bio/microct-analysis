"""DICOM intake with explicit geometry and intensity calibration. Arrays are Z,Y,X."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom


def grid_identity(meta):
    fields = {k: meta[k] for k in ["shape_zyx", "affine_xyz_to_world_mm", "sources"]}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def calibration(ds):
    """Require evidence for HU; do not equate a histogram threshold with calibration."""
    slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    kind = list(ds.get("ImageType", []))
    implicit_hu = (str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.2"
                   and len(kind) >= 3 and kind[0] == "ORIGINAL" and kind[2] != "LOCALIZER"
                   and ds.get("MultiEnergyCTAcquisition", "NO") == "NO")
    units = ds.get("RescaleType", "HU" if implicit_hu else None)
    if units != "HU" or not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("No valid positive-slope HU mapping in this CT header")
    result = dict(slope=slope, intercept=intercept, units="HU",
                  evidence="RescaleType" if "RescaleType" in ds else "DICOM CT Image Module implicit HU")
    # Resolve by private creator, rather than assuming its reserved block number.
    try:
        block = ds.private_block(0x0029, "SCANCO calibration")
    except KeyError:
        return result
    scaling, water = float(block[0].value), float(block[6].value)
    expected = 1000 / (scaling * water)
    if not np.isclose(slope, expected, rtol=1e-5) or not np.isclose(intercept, -1000):
        raise ValueError("DICOM HU mapping contradicts Scanco private calibration")
    result.update(mu_scaling=scaling, mu_water_per_cm=water,
                  private_hu_slope=expected, private_mapping_consistent=True)
    return result


def load_series(source, output):
    paths = sorted(Path(source).rglob("*.dcm"))
    if len(paths) < 2:
        raise ValueError("Need at least two DICOM slices")
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    first = headers[0]
    orient = np.array(first.ImageOrientationPatient, dtype=float).reshape(2, 3)
    if not np.allclose(orient @ orient.T, np.eye(2), atol=1e-6):
        raise ValueError("Invalid DICOM orientation cosines")
    normal = np.cross(*orient)
    positions = np.array([d.ImagePositionPatient for d in headers], dtype=float)
    order = np.argsort(positions @ normal)
    positions = positions[order]
    gaps = np.diff(positions @ normal)
    dz = float(np.median(gaps))
    if dz <= 0 or not np.allclose(gaps, dz, atol=1e-6, rtol=1e-4):
        raise ValueError("Duplicate/missing slices or nonuniform slice spacing")
    if not np.allclose(np.diff(positions, axis=0), dz * normal, atol=1e-5):
        raise ValueError("In-plane slice shifts: resampling needed before stacking")
    cal = calibration(first)
    for d in headers:
        if (d.SeriesInstanceUID != first.SeriesInstanceUID
                or (d.Rows, d.Columns) != (first.Rows, first.Columns)
                or not np.allclose(d.ImageOrientationPatient, orient.ravel())
                or not np.allclose(d.PixelSpacing, first.PixelSpacing)
                or calibration(d) != cal):
            raise ValueError("Inconsistent series, geometry, or calibration across slices")
    if len({str(d.SOPInstanceUID) for d in headers}) != len(headers):
        raise ValueError("Duplicate SOP instance UIDs")
    spacing = [float(first.PixelSpacing[1]), float(first.PixelSpacing[0]), dz]
    direction = np.column_stack([orient[0], orient[1], normal])
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = positions[0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "hu.npy"
    if dest.exists():
        raise ValueError(f"Refusing to overwrite {dest}; use a fresh run directory")
    volume = np.lib.format.open_memmap(dest, mode="w+", dtype="float32",
                                     shape=(len(paths), first.Rows, first.Columns))
    sources = []
    for z, idx in enumerate(order):
        p = paths[idx]
        d = pydicom.dcmread(p)
        volume[z] = d.pixel_array.astype(np.float32) * cal["slope"] + cal["intercept"]
        sources.append(dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    volume.flush()
    meta = dict(shape_zyx=list(volume.shape), spacing_xyz_mm=spacing,
                affine_xyz_to_world_mm=affine.tolist(), calibration=cal, sources=sources,
                coordinate_note="Array indices ZYX; affine consumes XYZ voxel-center indices",
                status="Calibrated from metadata; no independent phantom validation")
    meta["grid_id"] = grid_identity(meta)
    (output / "scan.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_scan(run):
    run = Path(run)
    meta = json.loads((run / "scan.json").read_text())
    meta["grid_id"] = grid_identity(meta)
    return np.load(run / "hu.npy", mmap_mode="r"), meta
