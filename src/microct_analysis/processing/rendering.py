"""Surface and slice rendering with camera control for landmarker agent.

Off-screen rendering primitives — renders images and returns file paths.
Does NOT interact with the agent directly.

COORDINATE CONVENTIONS:
- Volume data and positions.json: ZYX (SI=0, AP=1, ML=2)
- PyVista/VTK rendering: XYZ (ML=0, AP=1, SI=2)
- This module converts ZYX->XYZ for display meshes
- Camera params are in XYZ (PyVista convention)
- Functions accept ZYX vertices and convert internally
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.spatial import KDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _zyx_to_xyz(coords: np.ndarray) -> np.ndarray:
    """Convert ZYX coordinates to XYZ for PyVista."""
    return coords[:, ::-1] if coords.ndim == 2 else coords[::-1]


def _xyz_to_zyx(coords: np.ndarray) -> np.ndarray:
    """Convert XYZ coordinates back to ZYX."""
    return coords[:, ::-1] if coords.ndim == 2 else coords[::-1]


# ---------------------------------------------------------------------------
# 1. Segmentation validation
# ---------------------------------------------------------------------------


def validate_segmentation_for_landmarking(
    labels: np.ndarray,
    assignments: dict[str, int],
    workflow_checks: list[dict] | None = None,
) -> tuple[bool, str]:
    """Pre-flight check before any rendering or agent interaction.

    Checks:
    1. Each bone label has exactly 1 connected component (detect merge).
    2. Femur and tibia bounding boxes overlap <=20% of smaller bbox volume.
    3. No bone label touches volume boundary on >2 faces (detect FOV crop).

    Returns ``(ok, reason)``.  If not ok, stage aborts with confidence=low.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        return False, "labels must be a 3D array"

    # Check 1 -- connected components per label
    for bone_name, label_id in assignments.items():
        bone_mask = labels == label_id
        if not bone_mask.any():
            return False, f"label {label_id} ({bone_name}) has no voxels"
        _, n_components = ndimage.label(bone_mask)
        if n_components != 1:
            return (
                False,
                f"label {label_id} ({bone_name}) has {n_components} connected components (expected 1)",
            )

    # Check 2 -- bounding-box overlap between femur and tibia
    if "femur" in assignments and "tibia" in assignments:
        fem_mask = labels == assignments["femur"]
        tib_mask = labels == assignments["tibia"]
        fem_bbox = _bounding_box(fem_mask)
        tib_bbox = _bounding_box(tib_mask)
        overlap_vol = _bbox_overlap_volume(fem_bbox, tib_bbox)
        smaller_vol = min(_bbox_volume(fem_bbox), _bbox_volume(tib_bbox))
        if smaller_vol > 0 and overlap_vol / smaller_vol > 0.20:
            return (
                False,
                f"femur/tibia bounding boxes overlap {overlap_vol / smaller_vol:.0%} of smaller bbox (>20%)",
            )

    # Check 3 -- boundary touching (>2 faces)
    for bone_name, label_id in assignments.items():
        bone_mask = labels == label_id
        faces_touched = _count_boundary_faces(bone_mask)
        if faces_touched > 2:
            return (
                False,
                f"label {label_id} ({bone_name}) touches {faces_touched} volume boundary faces (>2, possible FOV crop)",
            )

    return True, "ok"


def _bounding_box(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_corner, max_corner) of a binary mask."""
    coords = np.argwhere(mask)
    return coords.min(axis=0), coords.max(axis=0)


def _bbox_volume(bbox: tuple[np.ndarray, np.ndarray]) -> float:
    """Volume (voxels) of a bounding box."""
    extent = bbox[1] - bbox[0] + 1
    return float(np.prod(extent))


def _bbox_overlap_volume(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
) -> float:
    """Intersection volume of two axis-aligned bounding boxes."""
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    extent = hi - lo + 1
    if np.any(extent <= 0):
        return 0.0
    return float(np.prod(extent))


def _count_boundary_faces(mask: np.ndarray) -> int:
    """Count how many of the 6 volume boundary faces a mask touches."""
    faces = 0
    if mask[0, :, :].any():
        faces += 1
    if mask[-1, :, :].any():
        faces += 1
    if mask[:, 0, :].any():
        faces += 1
    if mask[:, -1, :].any():
        faces += 1
    if mask[:, :, 0].any():
        faces += 1
    if mask[:, :, -1].any():
        faces += 1
    return faces


# ---------------------------------------------------------------------------
# 2. Session preparation
# ---------------------------------------------------------------------------


def prepare_landmark_session(
    segmentation_artifacts: dict[str, str],
) -> dict[str, Any]:
    """Load volumes, build meshes (render + snap), build KDTrees.

    Builds TWO meshes per bone:
    - render mesh: step_size=2 for display speed (~428K vertices)
    - snap mesh: step_size=1 for sub-voxel precision (~1.78M vertices)

    KDTrees are built on ZYX snap mesh vertices (matching positions.json coords).
    Meshes are cached per bone -- built once and reused across all landmarks.

    Returns dict with:
    - ``'labels'``: the label volume
    - ``'intensity'``: the intensity volume (or None)
    - ``'assignments'``: structure assignments dict  ``{name: label_id}``
    - ``'spacing'``: ``(z, y, x)`` spacing tuple
    - ``'meshes'``: ``{bone_name: {'render': pv.PolyData, 'snap_vertices': ndarray, 'snap_kdtree': KDTree}}``
    """
    import pyvista as pv

    # --- load volumes --------------------------------------------------------
    labels = _load_array(segmentation_artifacts.get("labels"))
    intensity_path = segmentation_artifacts.get("intensity") or segmentation_artifacts.get("filtered")
    intensity = _load_array(intensity_path) if intensity_path else None

    # --- load assignments ----------------------------------------------------
    assignments_path = segmentation_artifacts.get("structure_assignments")
    if assignments_path is None:
        raise ValueError("segmentation_artifacts must contain 'structure_assignments'")
    assignments_payload = json.loads(Path(assignments_path).read_text())
    # The file may wrap assignments in an outer dict
    if "assignments" in assignments_payload:
        raw = assignments_payload["assignments"]
    else:
        raw = assignments_payload
    # Normalise to {name: int}
    assignments: dict[str, int] = {str(k): int(v) for k, v in raw.items()}

    # --- spacing from metadata or default ------------------------------------
    metadata_path = segmentation_artifacts.get("metadata")
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    if metadata_path and Path(metadata_path).exists():
        meta = json.loads(Path(metadata_path).read_text())
        sp = meta.get("spacing") or meta.get("resampled_spacing") or meta.get("original_spacing")
        if sp is not None:
            spacing = tuple(float(v) for v in sp)  # type: ignore[assignment]

    # --- build meshes per bone -----------------------------------------------
    meshes: dict[str, dict[str, Any]] = {}
    for bone_name, label_id in assignments.items():
        bone_mask = (labels == label_id).astype(np.float32)
        if not bone_mask.any():
            continue

        # Render mesh -- step_size=2
        try:
            r_verts, r_faces, _, _ = marching_cubes(
                bone_mask, level=0.5, spacing=spacing, step_size=2,
            )
        except (ValueError, RuntimeError):
            # Fall back to step_size=1 if volume is too small for step_size=2
            r_verts, r_faces, _, _ = marching_cubes(
                bone_mask, level=0.5, spacing=spacing, step_size=1,
            )

        # Convert render mesh ZYX -> XYZ for PyVista
        r_verts_xyz = _zyx_to_xyz(r_verts)
        faces_pv = np.column_stack(
            [np.full(len(r_faces), 3), r_faces],
        ).ravel()
        render_mesh = pv.PolyData(r_verts_xyz, faces_pv)

        # Snap mesh -- step_size=1, stays in ZYX
        s_verts, _, _, _ = marching_cubes(
            bone_mask, level=0.5, spacing=spacing, step_size=1,
        )
        snap_tree = KDTree(s_verts)

        meshes[bone_name] = {
            "render": render_mesh,
            "snap_vertices": s_verts,
            "snap_kdtree": snap_tree,
        }

    return {
        "labels": labels,
        "intensity": intensity,
        "assignments": assignments,
        "spacing": spacing,
        "meshes": meshes,
    }


def _load_array(path: str | None) -> np.ndarray:
    """Load a numpy array from a NIfTI (.nii/.nii.gz) or .npy file."""
    if path is None:
        raise ValueError("path must not be None")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    suffix = "".join(p.suffixes).lower()
    if ".nii" in suffix:
        import nibabel as nib

        img = nib.load(str(p))
        return np.asarray(img.dataobj)
    if suffix == ".npy":
        return np.load(str(p))
    raise ValueError(f"unsupported file format: {suffix}")


# ---------------------------------------------------------------------------
# 3. Surface rendering
# ---------------------------------------------------------------------------


def render_surface_view(
    mesh: Any,  # pv.PolyData -- Any to avoid import at module level
    camera_position: tuple[float, float, float],
    focal_point: tuple[float, float, float],
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zoom: float = 1.0,
    resolution: tuple[int, int] = (1024, 1024),
    view_angle: float = 30.0,
    annotations: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """Render bone surface mesh from specified camera angle.

    Camera params are in XYZ (PyVista convention).
    Returns path to PNG screenshot.

    ``annotations``: list of dicts with:
      - ``'type': 'point'`` -- ``{'position': [x,y,z], 'color': str, 'label': str}``
      - ``'type': 'text'`` -- ``{'text': str, 'position': 'upper_left' | 'lower_left'}``
    """
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    plotter.add_mesh(mesh, color="ivory", smooth_shading=True, opacity=1.0)

    # Annotations
    if annotations:
        for ann in annotations:
            ann_type = ann.get("type", "")
            if ann_type == "point":
                pos = ann["position"]
                colour = ann.get("color", "red")
                label = ann.get("label", "")
                plotter.add_points(
                    np.array([pos]),
                    color=colour,
                    point_size=12,
                    render_points_as_spheres=True,
                )
                if label:
                    plotter.add_point_labels(
                        np.array([pos]),
                        [label],
                        font_size=14,
                        text_color=colour,
                        shape_opacity=0.0,
                    )
            elif ann_type == "text":
                text = ann.get("text", "")
                position = ann.get("position", "upper_left")
                plotter.add_text(text, position=position, font_size=12)

    # Camera
    plotter.camera.position = camera_position
    plotter.camera.focal_point = focal_point
    plotter.camera.up = view_up
    plotter.camera.view_angle = view_angle
    if zoom != 1.0:
        plotter.camera.zoom(zoom)

    # Screenshot
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix="surface_view_")
        import os

        os.close(fd)

    plotter.screenshot(output_path)
    plotter.close()
    return output_path


# ---------------------------------------------------------------------------
# 4. Slice rendering
# ---------------------------------------------------------------------------


def render_slice_view(
    volume: np.ndarray,
    slice_axis: int,
    slice_index: int,
    window: tuple[float, float] = (500.0, 800.0),
    mask: np.ndarray | None = None,
    annotations: list[dict] | None = None,
    resolution: tuple[int, int] = (1024, 1024),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: str | None = None,
) -> str:
    """Render 2D slice with windowing and optional mask overlay.

    Uses matplotlib for 2D slice rendering (avoids VTK thread issues).
    Windowing: ``center, width`` maps ``[center-width/2, center+width/2]``
    to ``[0, 255]``.

    Returns path to PNG screenshot.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")
    if not (0 <= slice_axis <= 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    if not (0 <= slice_index < volume.shape[slice_axis]):
        raise ValueError(
            f"slice_index {slice_index} out of range for axis {slice_axis} "
            f"with size {volume.shape[slice_axis]}"
        )

    # Extract slice
    slicing: list[Any] = [slice(None)] * 3
    slicing[slice_axis] = slice_index
    img = volume[tuple(slicing)].astype(np.float64)

    # Windowing
    center, width = window
    vmin = center - width / 2.0
    vmax = center + width / 2.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # Determine aspect ratio from spacing
    remaining_axes = [i for i in range(3) if i != slice_axis]
    pixel_spacing = (spacing[remaining_axes[0]], spacing[remaining_axes[1]])
    aspect = pixel_spacing[0] / pixel_spacing[1] if pixel_spacing[1] != 0 else 1.0

    dpi = 100
    fig_w = resolution[0] / dpi
    fig_h = resolution[1] / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(
        img, cmap="gray", norm=norm, aspect=aspect, interpolation="bilinear",
    )

    # Mask overlay -- contour lines
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_slice = mask_arr[tuple(slicing)]
        else:
            mask_slice = mask_arr
        if mask_slice.shape == img.shape:
            ax.contour(
                mask_slice.astype(float), levels=[0.5],
                colors=["lime"], linewidths=1.0,
            )

    # Annotations
    if annotations:
        for ann in annotations:
            ann_type = ann.get("type", "")
            if ann_type == "point":
                pos = ann.get("position", [0, 0])
                colour = ann.get("color", "red")
                label = ann.get("label", "")
                # position is (row, col) in slice image coords
                ax.plot(pos[1], pos[0], "o", color=colour, markersize=8)
                if label:
                    ax.annotate(
                        label,
                        (pos[1], pos[0]),
                        color=colour,
                        fontsize=10,
                        xytext=(5, -5),
                        textcoords="offset points",
                    )
            elif ann_type == "text":
                text = ann.get("text", "")
                position = ann.get("position", "upper_left")
                if position == "upper_left":
                    ax.text(
                        0.02, 0.98, text, transform=ax.transAxes,
                        fontsize=10, color="white", verticalalignment="top",
                        bbox=dict(
                            boxstyle="round", facecolor="black", alpha=0.5,
                        ),
                    )
                elif position == "lower_left":
                    ax.text(
                        0.02, 0.02, text, transform=ax.transAxes,
                        fontsize=10, color="white",
                        verticalalignment="bottom",
                        bbox=dict(
                            boxstyle="round", facecolor="black", alpha=0.5,
                        ),
                    )

    ax.set_axis_off()
    fig.tight_layout(pad=0)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(
            suffix=".png", prefix="slice_view_",
        )
        import os

        os.close(fd)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. Local geometry query
# ---------------------------------------------------------------------------


def query_local_geometry(
    point_zyx: tuple[float, float, float],
    mesh_vertices: np.ndarray,
    mesh_kdtree: KDTree,
    k: int = 12,
) -> dict:
    """Query local mesh geometry around a point.

    Returns::

        {
            'nearest_vertex': [z, y, x],
            'distance_mm': float,
            'neighbors': [[z,y,x], ...],
            'local_normal': [z, y, x],   # estimated from neighbor PCA
            'local_curvature': float,     # ML-direction spread of neighbors
        }
    """
    point = np.asarray(point_zyx, dtype=float)
    k_actual = min(k, len(mesh_vertices))

    distances, indices = mesh_kdtree.query(point, k=k_actual)
    if k_actual == 1:
        distances = np.array([distances])
        indices = np.array([indices])

    nearest_vertex = mesh_vertices[indices[0]].copy()
    neighbor_vertices = mesh_vertices[indices].copy()

    # Local normal via PCA of neighbor cloud
    centroid = neighbor_vertices.mean(axis=0)
    centered = neighbor_vertices - centroid
    if len(centered) >= 3:
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue -> normal direction
        normal = eigenvectors[:, 0].copy()
        # Orient normal outward (heuristic: point away from centroid)
        if np.dot(normal, point - centroid) < 0:
            normal = -normal
    else:
        normal = np.array([1.0, 0.0, 0.0])

    # Local curvature -- ML-direction spread (axis 2 = X in ZYX)
    ml_spread = float(np.std(neighbor_vertices[:, 2]))

    return {
        "nearest_vertex": nearest_vertex.tolist(),
        "distance_mm": float(distances[0]),
        "neighbors": neighbor_vertices.tolist(),
        "local_normal": normal.tolist(),
        "local_curvature": ml_spread,
    }
