# OA6-1RK analysis and review session

Session ID: **`01a15cdc-d75e-7ab1-bbcf-eb32010dc2be`**.

Recorded on 2026-10-21 from the active environment. `CODEX_SESSION_ID` and `CODEX_THREAD_ID` both contain this identifier. This document is a navigation record, not a copy of the conversation transcript.

## Preserved artifacts

Paths are relative to the repository root:

| Artifact | Location |
|---|---|
| Original calculated measurements | `data/python-protocol-run/final-measurements/measurements.json` |
| Original report | `data/python-protocol-run/final-measurements/REPORT.md` |
| Measurement provenance | `data/python-protocol-run/final-measurements/provenance.json` |
| Exact original-grid landmark coordinates and definitions | `data/python-protocol-run/review-10-measurements/measurement-spec.json` |
| Femoral endpoints and original sensitivity analysis | `data/python-protocol-run/review-10-measurements/femur-annotations.json` |
| Annotated 3D and tibial images | `data/python-protocol-run/review-10-measurements/` |
| Final candidate segmentation | `data/python-protocol-run/review-07-separated/labels.npy` |
| Anatomical coordinate transforms | `data/python-protocol-run/review-08-anatomical/frames.json` |
| Post hoc supplemental comparison | `data/python-protocol-run/supplement-comparison/` |
| Placement investigation, numerical sweeps, images and report | `data/python-protocol-run/placement-audit/` |
| Exported review-site assets | `data/python-protocol-run/web-review/` |

**The data directory is Git-ignored.** These artifacts are preserved on this machine, not included in Git commits or guaranteed to exist in a fresh clone. This session record is tracked separately. Preserve or back up the run directory to retain the full analysis.

The SHA-256 of the original `measurements.json` is:

```text
02b7cab6cdd7f90cec85ebe4b9441a3ffaf03ed074a783c1aae08ec9e0343242
```

It was checked against the hash recorded before the placement audit, in `supplement-comparison/comparison.json`, and matches. The original measurements were not replaced by supplemental values or audit alternatives.

## Findings to retain when resuming

Original provisional results: femoral length 2.422599 mm; width 3.393460 mm; width/length 1.400752; selected-section lateral tibial height 0.651 mm; patella core mineralized volume 1.275905 mm³. The tibial height is not established as the protocol's maximum-height measurement.

Supplemental Table S3, worksheet `Mint OA6,8 MMS`, row 32, matches OA6-1RK: primary-reader length 2.29 mm, width 3.48 mm, ratio 1.519651; second-reader ratio 1.404082. The sample metadata identifies WT, female, 12 months, 8 weeks after MMS. The original analysis did not infer that time point from the image.

The user initially prohibited supplemental access and later explicitly authorized it for comparison. The supplemental values were not used to choose the reported landmarks. Subsequent investigations are post-reference analyses and must not be described as blinded validation.

The placement audit found an unsupported restriction of both femoral width endpoints to arbitrary local W=0, and insufficient anatomical justification for the upper-groove midpoint. Notch C is effectively on the posterior surface, but exact notch placement remains uncertain. Physical transforms and distance arithmetic checked out. The original sensitivity ranges omit important placement uncertainty. No alternative audit value has been accepted as a corrected measurement.

Next scientific work: define and independently annotate the upper groove boundary and midpoint, posterior notch, and condylar edges without imposing a common transverse plane. Hide reference values during new observer annotation. Tibial growth-plate landmarks, maximum-height planes, and subchondral/trabecular ROIs remain unresolved.

An independent Astra agent, `/root/independent_protocol_review`, was launched without a conversation fork and reviewed measurements, registration, and placement limitations. Its checks do not constitute expert anatomical ground truth.

## Review site and implementation

Private Tailscale URL: <https://pop-os.tail852a76.ts.net:8443/>.

- Select `placement-audit` in the evidence-stage selector for audit images.
- [Placement audit report](https://pop-os.tail852a76.ts.net:8443/assets/evidence/placement-audit/REPORT.md).
- [Supplemental comparison report](https://pop-os.tail852a76.ts.net:8443/assets/evidence/supplement-comparison/REPORT.md).
- Notes entered in the site stay in that browser until exported; they do not update analysis files.
- The user service `microct-review.service` serves loopback port 8766. It is transient and needs recreation after reboot; Tailscale Serve persists separately.
- [Site source and restart instructions](../../skills/bone-morphometry/resources/review-site/README.md).
- Python workflow source: `skills/bone-morphometry/resources/python-protocol/`.

Relevant source commits: `1798a34` (Python protocol workflow) and `0a0f297` (3D review site). The run-specific datasets and artifacts are not part of those commits.

## 2026-10-22 presentation extension

The user supplied `/home/jimyao/gitrepos/.clipboard/uCT analysis parameters.pptx`
and requested identification and display of its parameters. All 23 numbered
parameters are recorded in `skills/bone-morphometry/resources/review-site/parameters.json`
and exposed on the review site with source slides and current status. Source
PPTX/PDF, slide PNGs and source checksum are in
`data/python-protocol-run/parameter-deck/` (Git-ignored).

The deck sharpens the femoral-width definition to lateral condylar ridge
midpoint versus medial condylar vertex. Existing width is flagged as a definition
mismatch rather than silently relabeled. No new accepted anatomical measurement
was generated by this cataloging step. Vertebral variants of parameters 15–23
require a different dataset. The question about cortical thickness at 100 or
500 µm below the growth plate remains unresolved, with both candidates displayed.

### IIOC segmentation follow-up, 2026-10-22

The user-marked region, new IIOC masks, image-traced growth-plate boundary,
measurements and multi-plane QC are preserved in `data/python-protocol-run/iioc/`.
See [IIOC review record](2026-10-22-oa6-1rk-iioc-review.md) for results, limits and
reproduction. Plate thicknesses remain directional candidates requiring lower
boundary review; the primary volume and height estimates remain provisional.
