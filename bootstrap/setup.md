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
uv run python -c "import microct_analysis; import pydicom; import nibabel; import numpy; import scipy; import skimage; import pyvista; import trame; print('analysis runtime ok')"
uv run python -c "import SimpleITK; print('SimpleITK ok')"
```

At session open, verify the workbench kernel can import the same runtime dependencies before spawning specialists:

```bash
jupyter-workbench exec "import microct_analysis; import pydicom; import nibabel; import numpy; import scipy; import skimage; import pyvista; import trame; import SimpleITK; print('analysis runtime ok')"
```

If any check fails, stop before running notebooks or skills. Install or link the missing package, then rerun all checks. Do not continue with partial bootstrap state because later notebook failures may look like analysis bugs instead of environment problems.

## Runtime module map

- Confidence gating: `microct_analysis.domain.confidence`
- Artifact handoff paths: `microct_analysis.domain.artifact_contracts`
- Workflow loading and validation: `microct_analysis.workflows.loading` and `microct_analysis.workflows.schema`
- Intake stage driver: `microct_analysis.stages.intake`
- Processing primitives: `microct_analysis.processing.{dicom,calibration,preprocess,threshold,segmentation,io,sanity,resample,orientation,surface,morphology}`
- Explain-then-apply helpers: `microct_analysis.workflows.explain`
- Plain-language feedback translation: `microct_analysis.workflows.feedback`

Stage drivers are executed through `jupyter-workbench exec --file` and may import only public `microct_analysis.processing` APIs. They must not import `jupyter_workbench.adapters.*`, internal workbench modules, or `mouse_ct` modules.

## Explain-then-apply protocol

Every domain correction requires a plain-language explanation before it runs. The explanation must say what will change, why the change addresses the review finding, and which artifact/component/parameter is affected. Use `microct_analysis.workflows.explain.correction_code()` to print and display the explanation before executing correction code, and keep the resulting record in the notebook.

Non-technical feedback is always translated before action. If the user says “that looks wrong,” points at “this area,” or supplies an annotated screenshot, use `microct_analysis.workflows.feedback.translate_visual_feedback()` or `translate_screenshot_feedback()` to convert the observation into inspectable domain operations. Do not require the user to name thresholds, masks, landmarks, or ROI parameters.

The notebook is the decision record. Preserve explanation cells, translated feedback, screenshots, stable component/landmark IDs, accepted parameter notes, confidence decisions, artifact paths, and loaded workflow identity so another agent can understand what changed without reopening the live scene.

## First-wedge workflow examples

Open and anchor one review session:

```bash
jupyter-workbench open --session-id microct-run-001
```

Run intake through the stage driver:

```bash
jupyter-workbench exec --session-id microct-run-001 --file src/microct_analysis/stages/intake.py -- /path/to/dicom-or-scan --output-dir .jupyter-workbench/microct-run-001
```

Run segmentation from intake metadata (inline snippet — `exec --file` does not pass arguments):

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

Poll durable visualization events:

```bash
jupyter-workbench exec --session-id microct-run-001 "$(uv run python - <<'PY'
from microct_analysis.workflows.review import event_poll_code
print(event_poll_code('microct-run-001', cursor=0, timeout=10.0))
PY
)"
```

Capture a review screenshot:

```bash
jupyter-workbench exec --session-id microct-run-001 "$(uv run python - <<'PY'
from microct_analysis.workflows.review import screenshot_code
print(screenshot_code('microct-run-001', '.jupyter-workbench', 'segmentation/screenshot_001.png'))
PY
)"
```

For cleanup-only work, no live visualization is required:

```bash
jupyter-workbench lineage microct-run-001
jupyter-workbench snapshot --session-id microct-run-001
```

Then use `microct_analysis.notebook_tasks.cleanup` helpers to identify dead-end cells and cells that preserve review decisions.
