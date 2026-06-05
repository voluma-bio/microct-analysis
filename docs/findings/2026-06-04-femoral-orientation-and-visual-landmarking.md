# Femoral landmarks are broken by orientation, not just the detectors — and visual landmarking may sidestep both

**Date:** 2026-06-04
**Source:** OA6-1RK validation + live 3D review (sessions `microct-oa6-1rk-003`, `-004`)
**Related work:** `landmark-detector-accuracy` (merged `58d137f`), `growth-plate-rework` (open)

## Summary

During live 3D review of OA6-1RK, every **femoral** landmark was visibly mis-placed
(uniformly — groove, notch, condylar edges). The cause is **upstream of the landmark
detectors**: the orientation stage only aligns the tibia, leaving the femur ~24°
askew. Separately, this exposed that **orientation barely matters if landmarks are
placed visually** — most measurements are rotation-invariant point-to-point
distances.

## Root cause: orientation aligns the tibia only

`stages/landmarks_orientation.py::_orient_tibia` derives the frame from the **tibia's**
PCA and applies that single frame to the whole volume. Measured long-axis angle to the
frame's SI axis (oriented_labels, session 003):

| Bone  | long-axis vector        | angle to SI axis |
|-------|-------------------------|------------------|
| Tibia | `[1.0, 0.0, 0.0]`       | **0.0°** (perfect) |
| Femur | `[0.915, -0.059, -0.399]` | **23.8°** (skewed) |

Every femoral detector assumes a *frontal-aligned* femur ("anterior-distal surface",
"most-superior point", "deepest midline concavity"), so a 24° skew misfires all of them.
The notch's "in-band" distal femoral length (2.057 mm) was a coincidence of distance,
not a correctly placed point. **Per earliest-wrong-input review discipline, orientation
is the fix point — not the femoral landmark detectors.**

## What actually depends on the frame

| Measurement | Frame-dependent? | Why |
|---|---|---|
| distal **femoral length** | No | point-to-point (groove→notch), rotation-invariant |
| trabecular BV/TV, Tb.Th, Tb.N, Tb.Sp (×3 ROIs) | No | ROI-internal stats |
| distal **femoral width** | **Yes** | `frontal_projected_width` — projects onto frame frontal plane |
| distal **femoral ratio** | **Yes** | = width ÷ length, inherits width |
| **tibial IIOC height** | Yes (tibia OK) | slice-count along SI; tibia already 0°-aligned |
| tibial width | Yes (tibia OK) | slice distance; tibia aligned |

## Implication: visual landmarking sidesteps orientation *and* the geometric detectors

If landmarks are placed from rendered views (human or vision model):
- **Identification is orientation-free** — anatomy is recognizable from any view.
- **Femoral length** and the **ratio numerator** become trivial point-to-point distances.
- The only frame-dependent femoral measurement (**width**, via `frontal_projected_width`)
  does **not** require re-orienting the volume — the frontal plane can be derived from the
  femoral landmarks themselves (groove + the two condylar edges).

This would retire the entire class of problems we have been fighting: the 24° femur skew,
`min(candidates)`, AP-recession notch scoring, and the fabricated growth-plate fixture —
all of which exist only to make axis-aligned geometric heuristics work.

## Open decision

1. **Femur-aware orientation** — add a femoral frontal frame to the orientation stage,
   then keep the geometric detectors. Smaller change; keeps current architecture.
2. **Visual landmarking** — render canonical views, place points by sight/VLM, derive
   frames + measurements from points. Larger change; retires orientation + detector
   brittleness.

## Reproduction

- Live review scene + the angle table: session `microct-oa6-1rk-003/landmarks/oriented_labels.npy`.
- Notch end-to-end validation (DFL 2.057 mm, growth plate still 247): session `-004`.
- The fabricated growth-plate fixture is `tests/fixtures/oa6_1rk_tibia_fill_ratios.npz`
  (test `test_growth_plate_oa6_1rk_selects_sustained_drop`, now `xfail`).
