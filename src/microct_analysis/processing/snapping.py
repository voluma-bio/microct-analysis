"""Pick-to-coordinate snap for both surface (3D) and slice (2D) domains.

Coordinate conventions
----------------------
- Volume data and positions.json : ZYX  (SI=0, AP=1, ML=2)
- PyVista / VTK rendering        : XYZ  (ML=0, AP=1, SI=2)
- ``snap_to_surface`` accepts camera params in XYZ, returns ZYX.
- ``snap_to_slice`` works entirely in ZYX.
- KDTrees are built on ZYX vertices.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; zero-safe."""
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Surface snapping (3-D, rendered view)
# ---------------------------------------------------------------------------


def snap_to_surface(
    pick_pixel: tuple[int, int],
    camera_params: dict[str, Any],
    render_mesh: Any,  # pv.PolyData in XYZ coords
    snap_kdtree: KDTree,
    snap_vertices: np.ndarray,  # ZYX coords
    image_resolution: tuple[int, int] = (1024, 1024),
) -> dict[str, Any]:
    """Map a 2D pixel pick on a rendered view to a precise 3D coordinate.

    Parameters
    ----------
    pick_pixel : (px, py)
        Pixel coordinates on the rendered image.
    camera_params : dict
        Must contain ``position``, ``focal_point``, ``view_up`` (all XYZ)
        and ``view_angle`` (degrees).
    render_mesh : pv.PolyData
        Mesh in XYZ coordinates used for ray-tracing.
    snap_kdtree : KDTree
        Built from *ZYX* vertices.
    snap_vertices : ndarray, shape (N, 3)
        The same ZYX vertices used to build *snap_kdtree*.
    image_resolution : (width, height)
        Resolution of the rendered image.

    Returns
    -------
    dict with keys ``coordinate`` (ZYX list), ``snap_distance_mm``,
    ``intersection_point`` (ZYX list or None), ``method``.
    """
    import pyvista as pv  # noqa: F811 -- local import to keep module light

    px, py = pick_pixel
    width, height = image_resolution

    camera_pos = np.asarray(camera_params["position"], dtype=float)
    focal = np.asarray(camera_params["focal_point"], dtype=float)
    view_up = np.asarray(camera_params["view_up"], dtype=float)
    view_angle = float(camera_params["view_angle"])  # degrees

    # ---- build ray direction in XYZ ----
    view_dir = _normalize(focal - camera_pos)
    right = _normalize(np.cross(view_dir, view_up))
    up = np.cross(right, view_dir)

    # NDC coords (-1..1)
    ndc_x = 2.0 * px / width - 1.0
    ndc_y = 1.0 - 2.0 * py / height

    # Scale by half-angle tangent & aspect ratio
    half_angle_tan = math.tan(math.radians(view_angle / 2.0))
    aspect = width / height
    ndc_x *= half_angle_tan * aspect
    ndc_y *= half_angle_tan

    ray_direction = _normalize(view_dir + ndc_x * right + ndc_y * up)
    far_distance = 10_000.0  # large enough to pass through any scene
    ray_end = camera_pos + far_distance * ray_direction

    # ---- ray-mesh intersection ----
    cleaned_mesh = render_mesh.clean()
    intersection_points, _cell_ids = cleaned_mesh.ray_trace(
        camera_pos.tolist(), ray_end.tolist()
    )

    if intersection_points is not None and len(intersection_points) > 0:
        hit_xyz = np.asarray(intersection_points[0], dtype=float)
        hit_zyx = hit_xyz[::-1]
        dist, idx = snap_kdtree.query(hit_zyx)
        snapped_zyx = snap_vertices[idx].tolist()
        return {
            "coordinate": snapped_zyx,
            "snap_distance_mm": float(dist),
            "intersection_point": hit_zyx.tolist(),
            "method": "ray_trace",
        }

    # ---- fallback: nearest vertex to ray line ----
    # Work in XYZ for the projection, then convert
    snap_vertices_xyz = snap_vertices[:, ::-1]  # ZYX -> XYZ

    v_to_origin = snap_vertices_xyz - camera_pos
    projections = np.dot(v_to_origin, ray_direction)
    closest_on_ray = camera_pos + projections[:, None] * ray_direction
    distances = np.linalg.norm(snap_vertices_xyz - closest_on_ray, axis=1)

    # Only consider vertices in front of camera
    mask = projections > 0
    if mask.any():
        distances[~mask] = np.inf
        nearest_idx = int(np.argmin(distances))
    else:
        # Nothing in front -- just pick the overall nearest
        nearest_idx = int(np.argmin(distances))

    snapped_zyx = snap_vertices[nearest_idx].tolist()
    snap_dist = float(distances[nearest_idx])

    return {
        "coordinate": snapped_zyx,
        "snap_distance_mm": snap_dist,
        "intersection_point": None,
        "method": "nearest_to_ray",
    }


# ---------------------------------------------------------------------------
# Slice snapping (2-D, axis-aligned slice)
# ---------------------------------------------------------------------------


def snap_to_slice(
    pick_slice_index: int,
    pick_pixel: tuple[int, int],
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    snap_mode: str = "center",
) -> dict[str, Any]:
    """Map a 2D slice pick to a precise voxel coordinate.

    Parameters
    ----------
    pick_slice_index : int
        Index along axis-0 (SI in ZYX).
    pick_pixel : (y, x)
        Pixel position on the 2D slice.
    mask : ndarray, shape (D, H, W), bool
        Binary mask in ZYX convention.
    spacing : (sz, sy, sx)
        Voxel spacing (unused in voxel-space output but reserved).
    snap_mode : str
        ``'center'`` -- centroid of mask on the picked slice.
        ``'edge_medial'`` -- minimum-ML (min x) voxel on the picked slice.
        ``'edge_lateral'`` -- maximum-ML (max x) voxel on the picked slice.
        ``'exact'`` -- use *pick_pixel* directly as (y, x).

    Returns
    -------
    dict with keys ``coordinate`` (ZYX floats in voxel space),
    ``snap_note``, ``on_mask``.
    """
    # Clamp slice index
    max_idx = mask.shape[0] - 1
    clamped = int(np.clip(pick_slice_index, 0, max_idx))

    slice_2d = mask[clamped]
    has_mask = bool(slice_2d.any())

    if snap_mode == "center":
        if has_mask:
            coords = np.argwhere(slice_2d)  # (N, 2) -> (y, x)
            mean_y = float(coords[:, 0].mean())
            mean_x = float(coords[:, 1].mean())
            on_mask = bool(slice_2d[int(round(mean_y)), int(round(mean_x))])
            return {
                "coordinate": [float(clamped), mean_y, mean_x],
                "snap_note": f"centroid of mask on slice {clamped}",
                "on_mask": on_mask,
            }
        else:
            py, px = pick_pixel
            return {
                "coordinate": [float(clamped), float(py), float(px)],
                "snap_note": f"no mask on slice {clamped}; using raw pick",
                "on_mask": False,
            }

    elif snap_mode == "edge_medial":
        if has_mask:
            coords = np.argwhere(slice_2d)
            min_x = int(coords[:, 1].min())
            # Among rows with min_x, pick the median y
            rows_at_min = coords[coords[:, 1] == min_x]
            median_y = float(np.median(rows_at_min[:, 0]))
            return {
                "coordinate": [float(clamped), median_y, float(min_x)],
                "snap_note": f"medial edge (min ML={min_x}) on slice {clamped}",
                "on_mask": True,
            }
        else:
            py, px = pick_pixel
            return {
                "coordinate": [float(clamped), float(py), float(px)],
                "snap_note": f"no mask on slice {clamped}; using raw pick",
                "on_mask": False,
            }

    elif snap_mode == "edge_lateral":
        if has_mask:
            coords = np.argwhere(slice_2d)
            max_x = int(coords[:, 1].max())
            rows_at_max = coords[coords[:, 1] == max_x]
            median_y = float(np.median(rows_at_max[:, 0]))
            return {
                "coordinate": [float(clamped), median_y, float(max_x)],
                "snap_note": f"lateral edge (max ML={max_x}) on slice {clamped}",
                "on_mask": True,
            }
        else:
            py, px = pick_pixel
            return {
                "coordinate": [float(clamped), float(py), float(px)],
                "snap_note": f"no mask on slice {clamped}; using raw pick",
                "on_mask": False,
            }

    elif snap_mode == "exact":
        py, px = pick_pixel
        on_mask = False
        if 0 <= py < slice_2d.shape[0] and 0 <= px < slice_2d.shape[1]:
            on_mask = bool(slice_2d[py, px])
        return {
            "coordinate": [float(clamped), float(py), float(px)],
            "snap_note": f"exact pick ({py}, {px}) on slice {clamped}",
            "on_mask": on_mask,
        }

    else:
        raise ValueError(f"Unknown snap_mode: {snap_mode!r}")
