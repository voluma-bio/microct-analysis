---
name: analysis-core
description: >
  Load when executing a micro-CT protocol: probe data, generate transparent
  Python, explain before execution, inspect artifacts, and avoid hidden
  scientific decisions.
type: mode-shift
---

# Analysis Core

Execute the accepted protocol with transparent, inspectable code. Use libraries
for mechanics; keep scientific choices visible.

## Execution Loop

1. Read the protocol and identify executable decisions.
2. Probe the actual data: dimensions, spacing, orientation, intensity range,
   labels, missing files, and obvious artifacts.
3. Explain the next meaningful step before running it.
4. Generate the smallest clear code that performs that step.
5. Run it and capture commands, stdout/stderr, paths, and versions when relevant.
6. Inspect the result with visual or quantitative QC.
7. Record provenance before moving on.

## Code Discipline

- Prefer simple scripts or notebooks over opaque frameworks.
- Use standard scientific libraries when they reduce boilerplate.
- Do not call a helper that hides thresholds, ROI definitions, formulas, or
  landmark choices from the reviewer.
- If using existing `microct_analysis` code, treat it as reference or mechanical
  support; state exactly which scientific choices it applies.
- Fix the earliest wrong input and rerun. Never patch reported measurements by
  post-hoc arithmetic.

Load `resources/execution-contract.md` when packaging an analyst result.
