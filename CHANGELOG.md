# Changelog

Caveman style. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Tibial IIOC trabecular ROI, same-grid TV/BV/BVTV, 3D sphere Tb.Th/Tb.Sp,
  explicitly model-derived Tb.N, input hashes, phantom checks and 3D/slice review.
- Provisional IIOC tissue/bone masks, original-grid volumes, explicit growth-plate
  traces, height conventions, width and directional plate candidates; multi-plane
  QC, sensitivity, independent placement review and opaque 3D cutaway controls.
- Presentation-derived catalog of 23 micro-CT parameters with source slides,
  anatomical context, per-parameter notes, explicit unresolved measurements,
  and vertebral data requirements on the private review site.
- OA6-1RK session record with session ID, preserved artifact paths, original
  measurement checksum, and outstanding placement-review findings.
- Private 3D review site with registered measurement rulers, 12-structure toggles,
  original/filtered/aligned slice browsing, evidence downloads, and exportable
  browser-local review notes; slice-orientation and request-boundary tests.
- Standalone OA-paper Python reference workflow: verified DICOM calibration,
  explicit filtering and watershed, reslicing, annotation-driven measurements,
  grid/hash-bound masks, and repeatable 3D/slice inspection artifacts.
- Numerical tests for calibration, physical geometry, ROI provenance, seed
  separation, and reslicing; documented protocol ambiguities and scope limits.

### Changed
- meridian-dev-workflow dependency unpinned: resolves to its latest release. The old `<0.14.0` range excluded current releases.
- Tibial IIOC heights and height/width ratio use direct endpoint distances after
  orientation correction; slice-count height retired from primary reporting.
- Review surfaces default to full opacity; hidden rulers and landmark labels
  are occluded by bone unless explicitly enabled.
- `@microct-study-design`: `opus46` fallback → `opus`. `@microct-analyst`: `sonnet5` fallback → `sonnet`.

## [0.1.1] - 2026-09-22

### Changed
- Removed Composer from microct-analyst's fallback policy; the base alias remains available during retirement.
- Study design and analyst now name Sol directly instead of the legacy gpt55 alias; Sol effort is explicit high.
- Code reviewer defaults to Astra for diversity from the Sol analyst, with Sol/Grok/Opus alternatives. DeepSeek is explicit-only, not an automatic scientific-review fallback.
- Requires Mars >=0.14.0 for the explicit-only review fallback policy.
- Requires meridian-dev-workflow >=0.13.6 for refreshed staffing and Muse aliases.
- `microct-study-design`, `microct-analyst`, and `microct-code-reviewer` can now use `structured-artifact`
  with `information-hierarchy` to produce reviewable static HTML artifacts for
  complex protocols, QC evidence, and results.
- Official micro-CT agent profiles now request `workspace-write` sandbox access so
  Meridian can add the work, archive, KB, project, and temp directories needed
  for normal package runs.
- Replaced the old six-agent stage pipeline with the converged Study Design → Analyst → Code Reviewer agent architecture.
- Added package skills for study design, analysis execution, provenance, environment probing, and domain-specific morphometry placeholders.
- Updated README and root AGENTS guidance for the new agent-as-thin-wrapper / skills-hold-knowledge model.

### Changed
- `render_surface_view` accepts and applies a `view_angle` (default 30°) so the rendered PNG's camera matches what `snap_to_surface`'s ray math uses; previously the angle was emitted but not honored.
- `microct-landmarker` agent profile rewritten for the CLI contract: `mct-landmark` subcommand calls replace `jupyter-workbench exec` invocations of Python primitives; viewpoint table, backstop retry protocol, cross-validation, and ROI sections preserved. Added a State-persistence section describing the `--cache <dir>` seam between subprocess calls.

- LLM writing quality pass across all agents and skill: collapsed repeated mct-visual-review recaps into one-line bridge sentences, deduplicated boundary rules that appeared in both operating contract and boundaries sections, split clause-heavy instruction blocks (analyst override matching, segmenter event-log translation, measurer upstream inputs, workflow-creator frontmatter list) into sub-bullets or tables, unified workflow-selection contract in analyst (analyst resolves from KB), replaced meta/governance phrasing with direct task language, normalized register (removed "The user talks to you", "Keep it anchored — alive —"), clarified "proceed silently" → "continue without pausing for review", replaced soft cleanup input labels ("keep/remove guidance", "final summary notes") with explicit artifact manifest table.
- Correct jupyter-workbench CLI examples for positional session commands.
- `bootstrap/setup/BOOTSTRAP.md`: add `Install` section with sibling-repo layout, `uv sync --extra dev`, and `[tool.uv.sources]` explanation; rename prior intro to `Verify environment`.
- `microct-analyst`: reframed description around the user-facing job (run an analysis) instead of "orchestrator" identity. Added an explicit workflow readiness gate — analyst will not pass a workflow with unresolved inferred or low/medium-confidence executable fields to specialists until the user reviews them.
- `microct-landmarker`: now the sole owner of ROI execution alongside landmarks and orientation. Added ROI artifacts (`roi_definitions`, `roi_masks`) to the stage report.
- `microct-measurer`: no longer runs the ROI stage driver. Consumes ROI artifacts from the landmarker; surfaces wrong ROI as evidence with a recommended pause instead of redefining it.
- `mct-visual-review` skill: now owns the generic semi-HITL policy shared across specialists — confidence semantics, explain-then-apply, plain-language feedback translation, reference image comparison, screenshot conventions, earliest-wrong-input correction, and the structured stage report shape. Removed duplicated policy from agent bodies.
- All agent bodies: stripped hardcoded Python helper module/function names and key bindings. Prompts stay at behavior boundaries (inputs, allowed tools, required outputs, escalation conditions). Stage names normalized to `segmentation`, `landmarks`, `measurements` across all stage reports.
- Femoral backstop rewritten as two-pass: `compute_backstop()` accepts an optional `femoral_frame` for Pass 2 cross-landmark validation; femoral coordinates are now physical mm throughout (no more double-scaling by spacing in DFL/condylar-width checks).
- `landmarks_orientation._rotation_matrix` orthogonalizes via Gram-Schmidt; rank-deficient input still produces a valid orthonormal matrix but logs a warning so downstream confidence aggregation can flag it.

### Added
- `mct-landmark` CLI (`microct_analysis.cli.landmark_ops`) — eight subcommands (`prepare`, `render-surface`, `render-slice`, `snap-surface`, `snap-slice`, `build-frame`, `backstop`, `emit`) that wrap the existing landmark primitives as JSON-in/JSON-out subprocesses, so the landmarker agent runs without a Jupyter kernel (no localhost-TCP — works under any sandbox).
- `processing/session_cache.py` — disk-backed session cache (per-bone render/snap mesh NPZs + spacing/assignments/source-volume paths in `session.json`); KDTree rebuilt on load, label/intensity volumes referenced by path and lazy-loaded only when the slice/tibial path needs them.
- `render-surface` → `snap-surface` camera-intrinsics round-trip: render emits the exact `camera_params` it used (`position`, `focal_point`, `view_up`, `view_angle`, `resolution`); snap reuses the JSON, deriving resolution from it by default.
- `tests/cli/test_landmark_ops_oa6_e2e.py` — acceptance gate proving the CLI loop reproduces the OA6-1RK result (correct notch `high` confidence; groove-notch swap rejected) inside the previously-failing codex/gpt-5.5 spawn sandbox, with an explicit assertion that no `jupyter`/`ipykernel`/`zmq` module is imported.

- Measurement subsystem: workflow-bound specs, geometry/volume/trabecular primitives, reporting payloads, stage driver, and override records.
- Real `jupyter-workbench derive` and `compact` cleanup handoff workflow.
- Explain-then-apply workflow helpers and skill protocol for feedback translation before corrections.
- Repo bootstrap with mars package, Python package, skills, and directory tree.
- First-wedge interactive workflow skills for segmentation review, landmark picking, ROI measurement, and notebook cleanup.
- Review helper code generators for jupyter-workbench event polling, screenshots, segmentation scenes, and landmark scenes.
- Cheap notebook cleanup heuristics for dead-end and review-decision cells.
- Femoral measurement frame derived from placed landmarks (lateral/medial condylar edges + groove + notch) — replaces per-signal axis guessing that failed on real OA6-1RK geometry. Two-pass backstop: frame-free signals catch gross errors (off-mesh, off-condyle, off-midline), frame-dependent signals catch groove-notch swap and AP/SI ordering. AP direction non-circularly verified by condylar-region mesh density with recession-differential fallback.
- Largest-component mesh preprocessing — multi-fragment femurs no longer inflate ML extremity onto shaft.
- `condylar_region_mask()` in `processing/surface.py` — single anatomical-region helper shared by frame construction and backstop.
- OA6-1RK acceptance test suite — 12 tests against real marching-cubes output gate femoral landmark correctness.
- `serialize_derived_frame()` writes `_derived_frame` to positions.json with all three axes (ml/ap/si), confidence, source landmarks, and AP verification method. `compute_frontal_projected_width` consumes unchanged.

### Removed
- Per-signal axis heuristics in the femoral backstop (`_detect_condylar_end`, `_detect_posterior_direction`, `si_in_condylar_band`, `posterior_position`) — failed on real diagonally-posed multi-component femurs; replaced by the landmark-anchored frame.
- `derive_ml_vector()` compatibility wrapper in `stages/visual_landmarks.py` — `emit_positions()` now builds and serializes the femoral frame directly.
- Two stale synthetic femoral backstop xfail tests — superseded by real-mesh OA6-1RK acceptance tests.
