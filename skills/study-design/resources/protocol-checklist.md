# Protocol Checklist

Use this shape for study-design artifacts.

## Research frame

- Research question
- Biological/material outcome
- Study groups or comparison axis
- Sample identity and scan inputs
- Required study blinding constraints

## Executable protocol

- Input volume format and spacing source
- Preprocessing steps and rationale
- Segmentation target and method
- Landmark or orientation frame, if needed
- VOI/ROI definitions with boundaries and exclusions
- Measurements with formulas, units, denominators, and required masks/labels
- QC views and quantitative acceptance checks
- Expected artifacts and directory shape

## Decisions

Classify each executable decision:

- `accepted`: explicitly provided or confirmed by the researcher
- `source-backed`: supported by cited protocol or fixture
- `assumed`: reasonable but must be surfaced before real execution
- `unknown`: blocks execution or requires redesign

## Analyst handoff

Name the accepted protocol artifact, input paths, constraints, anti-goals,
verification evidence, and stop/ask conditions. Do not pass raw conversation
history as the handoff.
