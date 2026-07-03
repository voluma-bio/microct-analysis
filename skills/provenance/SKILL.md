---
name: provenance
description: >
  Load when planning, running, or reviewing analysis provenance. Records protocol,
  techniques, execution steps, explain-before-execute judgments, artifacts,
  deviations, and reviewer evidence.
type: guardrail
---

# Provenance

Provenance is the audit trail that lets a reviewer understand what happened
without trusting hidden agent memory.

## Vocabulary

Use the package vocabulary consistently:

- **Protocol** — reusable or run-specific plan accepted before execution.
- **Technique** — planned method, aligned with PROV `Plan`.
- **Step** — executed activity, aligned with PROV `Activity`.
- **Judgment** — local explain-before-execute decision: why this step, these
  parameters, and this acceptance rule.
- **Analysis run** — workflow run that binds protocol, data, code, and outputs.
- **Artifact** — produced or consumed entity, aligned with PROV `Entity`.

## Minimum Record

For every meaningful step, record:

- input artifacts
- technique/protocol section used
- parameters and scientific choices
- command or code path
- output artifacts
- QC evidence
- judgment and confidence
- deviations and approval status

Load `resources/prov-model.md` when designing a manifest schema or reviewing
provenance completeness.
