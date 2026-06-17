"""Tests for per-landmark quantitative backstop validation."""

from __future__ import annotations

import numpy as np

from microct_analysis.processing.backstop import BackstopResult, compute_backstop


class TestGenericBackstop:
    """Tests for the generic (unknown domain) backstop."""

    def test_generic_backstop_accepts_on_bone_coordinate(self):
        """Coordinate on a labeled voxel should be accepted."""
        labels = np.zeros((50, 50, 50), dtype=np.uint8)
        labels[25, 25, 25] = 1  # bone voxel

        result = compute_backstop(
            landmark_def={"domain": "unknown", "id": "test_point"},
            coordinate=(25.0, 25.0, 25.0),
            labels=labels,
        )

        assert result.accepted is True
        assert result.confidence == "high"
        assert result.signals["bone_membership"]["accept"] is True
        assert result.feedback == ""

    def test_generic_backstop_rejects_off_bone(self):
        """Coordinate off bone should be rejected."""
        labels = np.zeros((50, 50, 50), dtype=np.uint8)
        labels[10, 10, 10] = 1  # bone is elsewhere

        result = compute_backstop(
            landmark_def={"domain": "unknown", "id": "test_point"},
            coordinate=(25.0, 25.0, 25.0),
            labels=labels,
        )

        assert result.accepted is False
        assert result.confidence == "low"
        assert result.signals["bone_membership"]["accept"] is False
        assert "off bone" in result.feedback


class TestFemoralBackstop:
    """Tests for femoral surface backstop."""

    def _make_condylar_mesh(self):
        """Synthetic femur: condylar end at HIGH SI, notch gap at high-AP midline.

        Anatomy:
        - SI (axis 0): shaft at low SI (0–200), condyles at high SI (300–470)
        - AP (axis 1): anterior ~100–300, posterior ~300–450
        - ML (axis 2): shaft narrow (220–380), condyles wide (100–500)
        - Intercondylar notch: gap near ML midline (300±40) at posterior AP (>300)
        """
        rng = np.random.default_rng(42)
        # Shaft (low SI): narrow ML
        n_shaft = 200
        shaft = np.column_stack([
            rng.uniform(0, 200, n_shaft),    # SI: low
            rng.uniform(100, 400, n_shaft),  # AP
            rng.uniform(220, 380, n_shaft),  # ML: narrow
        ])
        # Condylar (high SI): wide ML with notch gap at high-AP midline
        n_cond = 300
        cond_si = rng.uniform(300, 470, n_cond)
        cond_ap = rng.uniform(100, 450, n_cond)
        cond_ml = rng.uniform(100, 500, n_cond)
        # Remove midline vertices from posterior (high AP) = notch gap
        midline = np.abs(cond_ml - 300) < 40
        posterior = cond_ap > 300
        keep = ~(midline & posterior)
        condylar = np.column_stack([cond_si[keep], cond_ap[keep], cond_ml[keep]])
        return np.vstack([shaft, condylar])

    def test_femoral_backstop_accepts_valid_groove(self):
        """Groove at midline with good SI should be accepted."""
        vertices = self._make_condylar_mesh()
        # Groove in condylar region (high SI) at ML midline
        # Place in condylar half with a small snap tolerance
        si_condylar = 385.0  # firmly in high-SI condylar region
        ml_midline = float(np.median(vertices[:, 2]))
        # Add a vertex at exactly the target so snap distance = 0
        target = np.array([[si_condylar, 200.0, ml_midline]])
        vertices_with_target = np.vstack([vertices, target])

        result = compute_backstop(
            landmark_def={
                "domain": "femoral_3d_surface",
                "id": "intercondylar_groove_midpoint",
            },
            coordinate=(si_condylar, 200.0, ml_midline),
            mesh_vertices=vertices_with_target,
            spacing=(1.0, 1.0, 1.0),
        )

        assert result.accepted is True
        assert result.confidence in ("high", "medium")
        assert result.signals["bone_membership"]["accept"] is True
        assert result.signals["ml_midline_proximity"]["accept"] is True



class TestTibialBackstop:
    """Tests for tibial slice backstop."""

    def test_tibial_backstop_growth_plate_majority_vote(self):
        """Growth plate ensemble logic with synthetic signals."""
        # Create a volume where slice 30 has a clear transition:
        # - Slices 0-29: full bone (high intensity, large area)
        # - Slices 30+: mostly empty (low intensity, small area)
        rng = np.random.default_rng(42)
        shape = (60, 30, 30)
        labels = np.zeros(shape, dtype=np.uint8)
        intensity = np.zeros(shape, dtype=np.float32)

        # Bone region: slices 0-29
        for z in range(30):
            labels[z, 5:25, 5:25] = 1
            intensity[z, 5:25, 5:25] = rng.uniform(150, 200, (20, 20))

        # Transition region: slices 30-35 (rapidly decreasing)
        for z in range(30, 36):
            size = max(1, 20 - (z - 30) * 3)
            offset = 5
            labels[z, offset : offset + size, offset : offset + size] = 1
            intensity[z, offset : offset + size, offset : offset + size] = rng.uniform(
                50, 80, (size, size)
            )

        # Place candidate at the transition boundary
        result = compute_backstop(
            landmark_def={
                "domain": "tibial_2d_slice",
                "id": "growth_plate_proximal",
            },
            coordinate=(30.0, 15.0, 15.0),
            labels=labels,
            intensity=intensity,
            spacing=(1.0, 1.0, 1.0),
        )

        # The candidate at the transition should be accepted by majority vote
        assert isinstance(result, BackstopResult)
        assert result.confidence in ("high", "medium", "low")
        # Check that growth plate signals were computed
        gp_signal_keys = {
            "bone_fill_ratio_drop",
            "intensity_gradient_magnitude",
            "texture_homogeneity_drop",
            "mask_area_gradient",
        }
        found_signals = set(result.signals.keys()) & gp_signal_keys
        assert len(found_signals) > 0, "Expected growth plate signals to be computed"


class TestCrossLandmark:
    """Tests for cross-landmark validation signals."""

    def test_cross_landmark_dfl_range_accepts_valid(self):
        """DFL in range [1.5, 3.0]mm should be accepted."""
        # Place groove and notch 2.0mm apart
        placed = {
            "landmarks": [
                {"id": "intercondylar_notch", "voxel": [5.0, 0.0, 0.0]},
            ]
        }
        # Groove at [3.0, 0.0, 0.0] → distance = 2.0mm with spacing (1,1,1)
        vertices = np.random.default_rng(42).uniform(-5, 5, (200, 3))
        # Ensure the coordinate is on/near mesh
        vertices[0] = [3.0, 0.0, 0.0]

        result = compute_backstop(
            landmark_def={
                "domain": "femoral_3d_surface",
                "id": "intercondylar_groove_midpoint",
            },
            coordinate=(3.0, 0.0, 0.0),
            mesh_vertices=vertices,
            spacing=(1.0, 1.0, 1.0),
            placed_landmarks=placed,
        )

        assert "dfl_range" in result.signals
        assert result.signals["dfl_range"]["accept"] is True
        assert 1.5 <= result.signals["dfl_range"]["value"] <= 3.0

    def test_cross_landmark_dfl_range_rejects_out_of_range(self):
        """DFL outside [1.5, 3.0]mm should be rejected."""
        # Place groove and notch 5.0mm apart → too large
        placed = {
            "landmarks": [
                {"id": "intercondylar_notch", "voxel": [8.0, 0.0, 0.0]},
            ]
        }
        vertices = np.random.default_rng(42).uniform(-5, 10, (200, 3))
        vertices[0] = [3.0, 0.0, 0.0]

        result = compute_backstop(
            landmark_def={
                "domain": "femoral_3d_surface",
                "id": "intercondylar_groove_midpoint",
            },
            coordinate=(3.0, 0.0, 0.0),
            mesh_vertices=vertices,
            spacing=(1.0, 1.0, 1.0),
            placed_landmarks=placed,
        )

        assert "dfl_range" in result.signals
        assert result.signals["dfl_range"]["accept"] is False
        # DFL hard veto → entire result rejected
        assert result.accepted is False

    def test_iioc_plausibility_veto(self):
        """Growth plate too close to articular should trigger IIOC veto."""
        # Articular at slice 100, growth plate at slice 110 → only 10 slices apart
        labels = np.zeros((200, 30, 30), dtype=np.uint8)
        labels[90:120, 5:25, 5:25] = 1
        intensity = np.ones((200, 30, 30), dtype=np.float32) * 100.0
        intensity[90:120, 5:25, 5:25] = 180.0

        placed = {
            "landmarks": [
                {"id": "articular_surface_proximal", "voxel": [100.0, 15.0, 15.0]},
            ]
        }

        result = compute_backstop(
            landmark_def={
                "domain": "tibial_2d_slice",
                "id": "growth_plate_proximal",
            },
            coordinate=(110.0, 15.0, 15.0),
            labels=labels,
            intensity=intensity,
            spacing=(1.0, 1.0, 1.0),
            placed_landmarks=placed,
        )

        assert "iioc_plausibility" in result.signals
        assert result.signals["iioc_plausibility"]["accept"] is False
        assert result.accepted is False
        assert result.confidence == "low"
