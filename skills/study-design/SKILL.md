---
name: study-design
description: >
  Load when turning a micro-CT research question into an executable study
  protocol: outcome, anatomy, comparison, segmentation strategy, VOI/ROI,
  measurements, QC, and analyst handoff.
type: mode-shift
---

# Study Design

Design the study before designing the code. The output is a protocol that can
be executed and reviewed, not a list of interesting measurements.

## Core Loop

1. **Outcome** — What biological, material, or structural claim should this
   analysis support?
2. **Comparison** — What groups, regions, timepoints, sides, or compartments
   must be compared?
3. **Anatomy/material** — What structure is the unit of analysis, and where can
   segmentation or orientation fail?
4. **Measurements** — Which metrics answer the outcome, and which are tempting
   but irrelevant?
5. **VOI/ROI** — Define boundaries in anatomical or material terms before code
   terms.
6. **QC** — Decide what visual and quantitative evidence would reveal a wrong
   segmentation, wrong ROI, or wrong denominator.
7. **Handoff** — Write the analyst goal with evidence, constraints, anti-goals,
   and stop conditions.

## Discipline

Ask for the missing scientific decision rather than inventing it. If a default
is reasonable, label it as an assumption and require confirmation before first
use in a real analysis.

Load `resources/protocol-checklist.md` when drafting the protocol artifact.
