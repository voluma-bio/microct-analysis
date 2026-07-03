---
name: environment-setup
description: >
  Load before running micro-CT analysis code or model inference. Probe the actual
  Python, library, filesystem, CPU/GPU, and visualization environment before
  choosing execution details.
type: checkpoint
---

# Environment Setup

Probe first, then decide. Do not assume the machine has the libraries, GPU,
model weights, display stack, or writable paths implied by an old prompt.

## Required Probes

- Python executable and version
- package manager state (`uv`, virtualenv, installed project)
- key libraries needed by the protocol
- input data paths, sizes, and readability
- writable output directory
- CPU/GPU availability when runtime cost matters
- display/headless rendering support when QC images are required

## ML Segmentation

Do not ship a static decision tree for medical foundation models. If the
protocol needs ML segmentation, research current model docs and probe local
hardware at runtime. Choose the simplest credible loading path for this run and
record the model, weights, prompts, device, precision, and failure modes.

Load `resources/environment-checks.md` for a starter command checklist.
