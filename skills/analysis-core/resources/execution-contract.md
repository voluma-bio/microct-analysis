# Analyst Execution Contract

A reviewable analysis package includes:

- accepted protocol path or embedded protocol text
- data paths and checksums/sizes when practical
- generated code path and command used to run it
- environment evidence: Python version, key library versions, GPU/CPU facts if relevant
- measurement outputs with units and formulas
- QC artifacts with enough context to identify wrong segmentation, ROI, or orientation
- provenance manifest linking techniques, activities, judgments, and artifacts
- deviations from protocol with rationale and approval status
- unresolved risks and recommended reviewer focus

Do not mark the run complete if an expected artifact is missing or if a QC check
was skipped without explanation.
