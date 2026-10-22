# OA6-1RK: IIOC segmentation review

Session: `01a15cdc-d75e-7ab1-bbcf-eb32010dc2be`.

User request: segment the tibial secondary ossification center above the growth
plate, following the green marking in `Pasted Graphic 13.tiff`; measure whole
IIOC TV, BV, BV/TV, heights, plate thicknesses, and growth-plate-level width.

The local analysis is `data/python-protocol-run/iioc/`. It is Gitignored; masks
and screenshots are preserved locally, not backed up by this commit. The
[trace annotations](2026-10-22-oa6-1rk-iioc-annotations.json) and
[analysis code](../../skills/bone-morphometry/resources/python-protocol/iioc.py)
are tracked. Existing final measurements remain unchanged (SHA256
`02b7cab6cdd7f90cec85ebe4b9441a3ffaf03ed074a783c1aae08ec9e0343242`).

| Provisional measurement | Value |
|---|---:|
| Whole-IIOC TV | 3.45472 mm³ |
| Whole-IIOC BV | 2.32955 mm³ |
| BV/TV | 67.431% |
| Direct maximal height after tibial orientation correction | 0.92050 mm |
| Lateral midpoint height, V218 | 0.63315 mm |
| Medial midpoint height, V218 | 0.46358 mm |
| Widest screened section, manually refined chord V173 | 3.61548 mm |

Researcher revision: heights are direct Euclidean endpoint distances after tibial
orientation correction; slice counting is retired from primary results and
normalization. Direct maximal height / width is 0.25460. See the
[tibial trabecular follow-up](2026-10-22-oa6-1rk-tibial-trabecular-review.md)
for the separate trabecular compartment and six morphometry results.

The width is a maximal-width candidate, not an exhaustively validated maximum.
Plate first-run medians are 0.294 mm lateral and 0.399 mm medial, **not accepted
anatomical plate thicknesses**: trabecular bridges extend some runs, especially
medially. The lower plate boundary needs anatomical review. The site labels
both patches and results as directional candidates.

TV includes marrow and cortical shell; BV thresholds unfiltered calibrated HU
at 2459.875 inside the same original-grid ROI. This is not cancellous-only
BV/TV. Display bone comes from resliced median-filtered HU, so its mesh-label
volume differs from primary original-grid BV. ±2-voxel growth-face perturbations
give TV 3.312–3.597 mm³ and BV/TV 67.25–67.56% at the primary threshold. This is
method sensitivity, not a confidence interval or all annotation uncertainty.

Independent Astra review found three concrete problems, corrected before the
latest results: a prior label2 fragment against the femur affected onset height;
width used extrapolated edge values; a fixed-column medial search selected a
peripheral rim. The local articular-face correction, refined V173 edge trace,
and common V218 compartment section are recorded in the annotations. The first
marrow-fill failure and intermediate results are retained as `round1-*` and
`round3-results.json`. Remaining uncertainty includes manual GP interpolation,
ambiguous medial bridges, edge contours, and plate definition. Final independent review found no new coordinate/arithmetic
blocker for provisional reporting, but the small separate-in-section region near
U370,V200 needs anatomical ownership review before final TV/BV acceptance.

Review at <https://pop-os.tail852a76.ts.net:8443/?parameter=9>. Parameters 7–14
link results, masks and QC. Parameter 12 links all height rulers. IIOC tissue,
bone and plate patches have separate 3D toggles and aligned slice overlays.
Tibial views and optional opaque cutaway permit inspection without transparency.
Notes remain browser-local and can be exported. Code, annotations, full results,
sensitivity and image evidence are in the site's `iioc` evidence stage.

Reproduce from the repository root (requires existing input run and web assets):

```bash
mkdir -p data/python-protocol-run/iioc
cp docs/findings/2026-10-22-oa6-1rk-iioc-annotations.json data/python-protocol-run/iioc/annotations.json
MPLCONFIGDIR=/tmp/microct-mpl .venv/bin/python skills/bone-morphometry/resources/python-protocol/iioc.py data/python-protocol-run
.venv/bin/python skills/bone-morphometry/resources/review-site/export.py data/python-protocol-run data/python-protocol-run/web-review --study-pdf '../microct-arena/datasets/OA-uCT paper 251123.pdf'
```

Validation: 14 Python tests (including inverse ROI registration, curved growth
face and marrow/exterior-concavity phantom), parameter-panel checks, numerical
mask/volume invariants, opaque anterior/posterior browser screenshots, registered
height selection, and all four live slice-overlay endpoints.
