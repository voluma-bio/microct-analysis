# microct-analysis bootstrap

## Install

`microct-analysis` depends on `jupyter-workbench` as an editable sibling package. The expected layout:

```text
parent/
  jupyter-workbench/   ← git checkout of jupyter-workbench
  microct-analysis/    ← this repo
```

With that layout in place, install from within this repo:

```bash
uv sync --extra dev
```

`pyproject.toml` resolves `jupyter-workbench` via `[tool.uv.sources]` relative paths; no manual path edits are needed. If the sibling is missing, `uv sync` will error — clone it first. Processing is self-contained in `microct_analysis.processing` and no longer imports `mouse-ct`.

## Verify environment

Run these checks after creating or updating the environment:

```bash
uv run python -c "from jupyter_workbench import SessionService, ExecutionService, SnapshotService; print('workbench ok')"
uv run python -c "import microct_analysis; import pydicom; import nibabel; import numpy; import scipy; import skimage; print('analysis runtime ok')"
uv run python -c "import SimpleITK; print('SimpleITK ok')"
```

Visualization libraries are protocol-dependent. Probe them only when the accepted protocol requires rendered QC:

```bash
uv run python - <<'PY'
import importlib.util
for name in ['matplotlib', 'pyvista', 'trame']:
    print(f'{name}: {bool(importlib.util.find_spec(name))}')
PY
```

If any required check fails, stop before running analysis. Install or link the missing package, then rerun the relevant checks. Do not continue with partial bootstrap state because later failures may look like analysis bugs instead of environment problems.

## Current agent flow

The package flow is:

```text
microct-study-design → microct-analyst → microct-code-reviewer
```

1. `microct-study-design` produces an executable protocol and analyst goal.
2. `microct-analyst` applies the readiness gate, probes the environment/data, generates transparent analysis code, records provenance, and packages artifacts.
3. `microct-code-reviewer` compares protocol, code, outputs, QC, and provenance before results are trusted.

Do not route new work through the retired segmenter/landmarker/measurer/cleanup agent pipeline. Stage-driver examples below are legacy runtime references only.

## Runtime module map

These modules remain as test-covered runtime/reference material while the prompt package moves toward transparent generated analysis code:

- Confidence gating: `microct_analysis.domain.confidence`
- Artifact handoff paths: `microct_analysis.domain.artifact_contracts`
- Workflow loading and validation: `microct_analysis.workflows.loading` and `microct_analysis.workflows.schema`
- Intake stage driver: `microct_analysis.stages.intake`
- Processing primitives: `microct_analysis.processing.{dicom,calibration,preprocess,threshold,segmentation,io,sanity,resample,surface,morphology}`
- Explain-then-apply helpers: `microct_analysis.workflows.explain`
- Plain-language feedback translation: `microct_analysis.workflows.feedback`

Runtime helpers may import only public `microct_analysis.processing` APIs. They must not import `jupyter_workbench.adapters.*`, internal workbench modules, or `mouse_ct` modules.

## Explain-before-execute protocol

Every scientifically meaningful correction or irreversible step needs a plain-language explanation before it runs. The explanation must say what will change, why it addresses the protocol or QC finding, and which artifact/component/parameter is affected.

Non-technical feedback is translated before action. If the user says “that looks wrong,” points at an area, or supplies an annotated screenshot, convert the observation into inspectable domain operations. Do not require the user to name thresholds, masks, landmarks, or ROI parameters.

The notebook or run directory is the decision record. Preserve explanation cells, translated feedback, screenshots, stable IDs, accepted parameter notes, confidence decisions, artifact paths, and loaded protocol identity so a reviewer can understand what changed without reopening a live scene.

## Legacy runtime examples

These commands are examples for preserved runtime modules, not the current agent architecture.

Open and anchor one review session:

```bash
jupyter-workbench open --session-id microct-run-001
```

Run intake through the stage driver:

```bash
jupyter-workbench exec --session-id microct-run-001 --file src/microct_analysis/stages/intake.py -- /path/to/dicom-or-scan --output-dir .jupyter-workbench/microct-run-001
```

Run segmentation from intake metadata:

```bash
jupyter-workbench exec --session-id microct-run-001 "$(uv run python - <<'PY'
import json
from microct_analysis.stages.segmentation import run_segmentation
print(json.dumps(run_segmentation(
    intake_metadata_path='.jupyter-workbench/microct-run-001/intake/volume_metadata.json',
    output_dir='.jupyter-workbench/microct-run-001',
)))
PY
)"
```

Capture lineage/snapshots for review or cleanup reference:

```bash
jupyter-workbench lineage microct-run-001
jupyter-workbench snapshot --session-id microct-run-001
```
