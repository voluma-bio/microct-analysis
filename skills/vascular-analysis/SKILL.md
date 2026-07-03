---
name: vascular-analysis
description: >
  Load when a micro-CT protocol concerns vascular networks, perfusion casts,
  angiography contrast, vessel segmentation, branching, diameter, density, or
  connectivity.
type: reference
model-invocable: true
---

# Vascular Analysis

This is a domain placeholder, not a complete protocol library. Do not reuse bone
morphometry assumptions for vessels.

When vascular analysis is in scope, define:

- contrast mechanism and expected intensity relationship to background
- vessel scale range and resolution limits
- segmentation method and leakage/fill artifacts
- skeletonization or centerline approach, if needed
- metrics: vessel volume fraction, diameter distribution, branch count,
  connectivity, tortuosity, perfused region, and spatial density
- QC: overlays, false-positive regions, disconnected components, and diameter
  sanity checks

If the protocol requires detailed vascular domain rules not present here, stop
and research or ask before execution.
