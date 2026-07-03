# Provenance Model

A compact JSON/YAML manifest can use this shape:

```yaml
analysis_run:
  id: ...
  protocol: ...
  data: []
techniques:
  - id: segment-bone
    protocol_ref: segmentation.method
    description: ...
steps:
  - id: step-001
    technique: segment-bone
    command: ...
    inputs: []
    outputs: []
judgments:
  - id: judgment-001
    step: step-001
    question: Why is this parameter/step acceptable?
    rationale: ...
    confidence: high|medium|low
    qc_evidence: []
artifacts:
  - id: ...
    path: ...
    kind: volume|mask|table|figure|script|notebook|manifest
    derived_from: []
deviations:
  - protocol_ref: ...
    original: ...
    actual: ...
    rationale: ...
    approval: accepted|unreviewed|rejected
```

Reviewer rule: if a result number cannot be traced through artifact → step →
technique/protocol → judgment, provenance is incomplete.
