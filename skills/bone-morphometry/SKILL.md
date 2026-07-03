---
name: bone-morphometry
description: >
  Load when the protocol concerns bone structure, osteoarthritis, subchondral
  bone, trabecular/cortical morphometry, landmarks, VOIs, or BV/TV-style
  measurements.
type: reference
model-invocable: true
---

# Bone Morphometry

Bone analysis starts with the biological compartment and comparison, not the
metric acronym. Define the bone, side/region, VOI, segmentation target, and
acceptance checks before computing measurements.

## Common Decisions

- Bone and compartment: femur, tibia, subchondral plate, epiphysis, metaphysis,
  cortical shell, trabecular compartment.
- Comparison: medial vs lateral, injured vs contralateral, treatment vs control,
  anterior/posterior, depth from articular surface.
- VOI: anatomical boundaries, offsets, excluded growth plate or cortical shell,
  spacing-aware dimensions.
- Segmentation: bone threshold, label cleanup, connected components, partial
  volume risk, metal/contrast artifacts.
- Measurements: BV/TV, tissue volume, bone volume, trabecular thickness,
  trabecular number, trabecular separation, cortical thickness, subchondral plate
  thickness, BMD/TMD if calibration supports it.

## QC

Show enough evidence to catch wrong-side, wrong-depth, wrong-threshold, and
wrong-denominator failures. A metric table without ROI overlays is not enough.

Load `resources/subchondral-bone.md` for osteoarthritis mouse-knee design cues.
