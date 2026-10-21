"""Inspection artifacts: unsmoothed surfaces, fixed recorded cameras, slice overlays."""

import json
import colorsys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

COLORS = ["#e3bb78", "#56b4e9", "#cc79a7", "#009e73", "#d55e00", "#f0e442", "#999999"]


def label_color(value):
    if value <= len(COLORS):
        return COLORS[value-1]
    return matplotlib.colors.to_hex(colorsys.hsv_to_rgb((value * 0.61803398875) % 1, 0.65, 0.72))


def render_review(hu, labels, meta, output, names=None, display_labels=None):
    """Rendering samples surfaces every two voxels; calculations retain full resolution."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = names or {}
    affine = np.array(meta["affine_xyz_to_world_mm"])
    meshes = []
    boxes = ndi.find_objects(labels)
    for value, box in enumerate(boxes, 1):
        if display_labels is not None and value not in display_labels:
            continue
        if box is None:
            continue
        mask = labels[box] == value
        if np.count_nonzero(mask) < 8:
            continue
        vertices, faces, _, _ = marching_cubes(np.pad(mask, 1), level=0.5, step_size=2)
        vertices += np.array([s.start for s in box]) - 1
        xyz = vertices[:, ::-1] @ affine[:3, :3].T + affine[:3, 3]
        mesh = pv.PolyData(xyz, np.column_stack([np.full(len(faces), 3), faces]).ravel())
        mesh.save(output / f"label-{value}.vtp")
        meshes.append((value, mesh))
    if not meshes:
        raise ValueError("Nothing to render")
    plot = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    plot.set_background("#f5f5f2")
    for value, mesh in meshes:
        plot.add_mesh(mesh, color=label_color(value), smooth_shading=False,
                      label=names.get(str(value), f"label {value}"))
    if len(meshes) <= 10:
        plot.add_legend(bcolor="white", size=(0.27, 0.25))
    plot.add_axes()
    center = np.array(plot.center)
    radius = max(plot.bounds[1]-plot.bounds[0], plot.bounds[3]-plot.bounds[2],
                 plot.bounds[5]-plot.bounds[4]) * 2.5
    cameras = {}
    for name, vector in [("plus-y", [0, 1, 0]), ("minus-y", [0, -1, 0]),
                         ("plus-x", [1, 0, 0]), ("oblique", [1, 1, 0.6])]:
        vec = np.array(vector) / np.linalg.norm(vector)
        plot.camera_position = [center + radius * vec, center, [0, 0, 1]]
        plot.enable_parallel_projection()
        plot.reset_camera()
        plot.add_text(f"{name} | display mesh step=2; no surface smoothing", name="caption", font_size=12)
        plot.screenshot(output / f"{name}.png")
        cameras[name] = [list(v) for v in plot.camera_position]
    plot.close()
    (output / "cameras.json").write_text(json.dumps(dict(cameras=cameras,
        displayed_labels=[value for value, _ in meshes], display_mesh_step_voxels=2,
        surface_smoothing=False), indent=2) + "\n")
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    maximum = max(2, int(labels.max()))
    cmap = ListedColormap([label_color(i) for i in range(1, maximum+1)])
    norm = BoundaryNorm(np.arange(0.5, maximum+1.5), cmap.N)
    for axis in range(3):
        for col, fraction in enumerate([0.3, 0.5, 0.7]):
            index = int((hu.shape[axis]-1) * fraction)
            plane = np.take(hu, index, axis=axis)
            lab = np.take(labels, index, axis=axis)
            ax = axes[axis, col]
            ax.imshow(plane, cmap="gray", vmin=-1000, vmax=6000, origin="lower")
            ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=cmap, norm=norm, alpha=0.35,
                      origin="lower", interpolation="nearest")
            ax.set_title(f"{'ZYX'[axis]} index {index}")
    fig.tight_layout()
    fig.savefig(output / "slices.png", dpi=140)
    plt.close(fig)
