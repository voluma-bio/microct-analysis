---
name: mct-visual-review
description: >
  Load when reviewing or correcting micro-CT analysis output inside a
  jupyter-workbench session. Owns the generic semi-HITL loop: confidence
  assignment, explain-then-apply correction, plain-language feedback
  translation, reference image comparison, screenshot conventions,
  earliest-wrong-input correction, and the structured stage report shape
  every specialist returns.
---

# MCT Visual Review

Generic semi-HITL review loop shared by every micro-CT analysis specialist.
Stage-specific anatomy, protocol constants, and execution sequence come from
the specialist agent and the workflow file.

## Mechanics

### Screenshot capture

- Capture screenshots at decision points: stage completion, before/after
  corrections, ambiguity surfaces, and low-confidence pauses.
- Path convention: `<stage>/screenshot_<NNN>.png` with zero-padded NNN
  incrementing within the stage. Stage names are the canonical short
  identifiers (`segmentation`, `landmarks`, `measurements`).
- Reference screenshots in the stage report's `artifacts.screenshots`
  list.

### Event polling

- Poll the workbench event log from a caller-owned cursor so each event is
  processed once.
- Translate generic UI events into the specialist's domain operations;
  do not redefine the event contract inside a specialist.
- Resolve visual selections through stable component, landmark, ROI, or
  measurement IDs from artifacts rather than actor order or color alone.

### Scene refresh

- After each accepted correction, rerun the smallest pipeline step needed
  and refresh only the affected scene state.
- Preserve stable IDs where possible so before/after explanations remain
  traceable.
- Summarize what changed and what the reviewer should now see in the
  refreshed scene.

### Per-landmark viewpoint selection

When the specialist defines per-landmark primary views (e.g., a viewpoint
table mapping landmark IDs to camera angles), always render the primary
view first. Anatomical features can be occluded or mimicked by nearby
structures from the wrong angle — rendering generic views wastes the view
budget and risks mis-identification. Secondary views supplement when the
primary is inconclusive; they do not replace it.

### Reference image comparison

Each stage has reference images attached to the workflow. Use them — do
not rely on memory or general anatomy knowledge when a workflow reference
exists.

- Load only the stage-relevant references the analyst passed.
- Compare the current scene or screenshot against each reference for the
  visual checks the workflow defines (component count, anatomical
  separation, edge quality, label placement, ROI position).
- Record comparison observations in plain language in the stage report.
- A reference mismatch is confidence evidence, not a silent override.

## Policy

### Confidence assignment

Every stage ends with a confidence rating that drives the analyst's gate.

- **`high`** — workflow targets and visual evidence agree across acceptance
  checks and reference images. Record the result in the notebook and
  continue to the next step without pausing for review.
- **`medium`** — output is usable but a real concern remains: a sanity
  warning, a usable threshold deviation, an override applied with
  rationale. Proceed and flag in the report so the analyst surfaces it.
- **`low`** — multiple plausible corrections, blocked interpretation,
  ambiguous bone identity, missing upstream artifact, or evidence that
  contradicts the workflow. Pause, attach evidence (screenshots, current
  state, reference comparison), and let the analyst route to the user.

Specialists assign stage confidence and recommend an action. The analyst
alone decides run-level proceed / flag / pause.

### Explain then apply

Before any corrective mutation — threshold change, seed reassignment,
landmark move, ROI shift, override — state in plain language:

- the concrete domain anchor being changed (named landmark, named ROI,
  named threshold, specific component)
- the current value or assignment
- the proposed value or assignment
- the workflow or reference evidence that supports the change
- the physical and visual consequence the reviewer should see after rerun

Then apply with the smallest rerun the change requires. After execution,
summarize what changed and what the reviewer should now see in the scene.

### Plain-language feedback handling

Users describe problems in non-technical terms — screenshots, "the bone on
the left looks wrong," "this slice is rotated." Translate that into domain
operations before asking the user to name parameters, masks, landmarks, or
anatomy jargon.

Inspect the current scene first. If the feedback maps to a clear domain
operation, propose it via explain-then-apply and ask for confirmation. Ask
for jargon only when the visual evidence is genuinely ambiguous and you
cannot narrow the operation without a technical disambiguator.

### Escalation rules

- Use earliest-wrong-input correction. When a finding implicates an
  upstream artifact, fix the earliest wrong input before redoing
  downstream work. Do not patch downstream numbers, overlays, or labels to
  mask an upstream problem.
- Order: intake/orientation → segmentation/structure-ID → landmarks → ROI
  → measurement.
- If the wrong input lives in a prior specialist's stage, stop and surface
  that to the analyst rather than working around it.
- Pause with `low` confidence when evidence is ambiguous, contradictory,
  or blocked by a missing upstream artifact.

### Structured stage report

Every specialist returns this shape to the analyst:

```json
{
  "stage": "<segmentation|landmarks|measurements>",
  "confidence": "high|medium|low",
  "evidence": "plain-language summary of what the confidence rests on",
  "recommended_action": "proceed|flag|pause",
  "artifacts": { "<key>": "<session-relative path>", "screenshots": ["..."] }
}
```

- `evidence` cites the workflow checks, reference comparisons, and sanity
  signals the rating draws from. Vague "looks fine" is not evidence.
- `recommended_action` is a recommendation — the analyst decides.
- `artifacts` lists the stage's durable outputs by stable key, with paths
  relative to the session artifact directory.

Per-stage artifact keys are defined in each specialist's prompt.

## Out of scope

Do not look here for:

- session lifecycle or spawn sequencing — see the analyst prompt
- anatomy knowledge or protocol constants — see the workflow file
- stage execution order or driver invocation — see specialist prompts
- domain operations specific to one stage (seed curation, landmark
  placement, ROI offsets, measurement formulas) — see specialist
  prompts and the workflow file
