# Tibial IIOC trabecular analysis and direct-height revision

Session: `01a15cdc-d75e-7ab1-bbcf-eb32010dc2be`.

The researcher clarified that these measurements concern **tibial bone**, requested
a separate trabecular IIOC segmentation, and explicitly replaced slice-count
height with direct measurement after correction of tibial orientation.

The new compartment is anatomically traced below the subchondral plate and
inside the endocortical boundary, with marrow retained in TV. The earlier whole
IIOC volumes include shell and plate and must not substitute for these results.
Boundary exclusion is provisional, especially where the medial compartment is
dense continuous bone. This narrower ROI differs from the paper's description
including all noncortical solid/trabecular tissue through parts of the growth
plate region.

| Tibial trabecular metric | Provisional result |
|---|---:|
| TV | 1.30764 mm³ |
| BV | 0.52380 mm³ |
| BV/TV | 40.0567% |
| Tb.Th | 97.59 µm |
| Tb.N, model-derived | 4.10440 mm⁻¹ |
| Tb.Sp | 262.91 µm |

All six results use the same aligned, median-filtered and linearly resliced HU
segmentation at >2459.875 HU and 10.5 µm voxels. Tb.Th and Tb.Sp are averages of
largest-contained-sphere diameter maps over bone and ROI marrow respectively.
Radii are quantized at 0.5 voxels. Tb.N = (BV/TV)/Tb.Th with thickness in mm; it
is model-derived and not direct Scanco trabecular number. See the generated
[report](../../data/python-protocol-run/tibial-trabecular/REPORT.md) for method
references and the complete revised height table.

Primary heights are direct Euclidean distances between saved endpoints after
shaft-based tibial orientation correction: maximum 0.92050 mm, medial 0.46358 mm,
lateral 0.63315 mm. The direct maximum / growth-plate-width ratio is 0.25460.
The slice-count value is retired from primary reporting, the site catalog and
normalization. Its earlier result remains explicitly historical in the JSON
provenance rather than being silently relabeled as a direct distance.

Independent Astra review reproduced the arithmetic and checked sphere propagation
against brute-force enumeration. It identified the dense medial anatomical
boundary as unresolved and required hashes of actual consumed arrays; those
hashes are now captured before loading inputs. Excluding one sampling layer
changes Tb.Th 97.59→102.95 µm and Tb.Sp 262.91→270.30 µm. This is a change in sampled
population, not a correction or full estimate of sphere-clipping bias.

Preserved local artifacts (Gitignored):
`data/python-protocol-run/tibial-trabecular/` contains the tissue/bone/marrow
masks, cropped diameter maps with index offsets, full results and sensitivity,
coronal/sagittal QC, mesh files, source copy and hashes. The tracked
[annotations](2026-10-22-oa6-1rk-tibial-trabecular-annotations.json) and
[analysis script](../../skills/bone-morphometry/resources/python-protocol/tibial_trabecular.py)
allow reproduction from the existing aligned run:

```bash
mkdir -p data/python-protocol-run/tibial-trabecular
cp docs/findings/2026-10-22-oa6-1rk-tibial-trabecular-annotations.json data/python-protocol-run/tibial-trabecular/annotations.json
MPLCONFIGDIR=/tmp/microct-mpl .venv/bin/python skills/bone-morphometry/resources/python-protocol/tibial_trabecular.py data/python-protocol-run
.venv/bin/python skills/bone-morphometry/resources/review-site/export.py data/python-protocol-run data/python-protocol-run/web-review --study-pdf '../microct-arena/datasets/OA-uCT paper 251123.pdf'
```

The site has separate trabecular tissue and mineralized-bone surfaces, separate
aligned slice overlays, and a dedicated six-metric section. Parameters 15–20 in
the original deck still describe the tibial shaft ROI; they are deliberately not
repurposed as epiphyseal trabecular results. Parameter 12 now shows the direct
maximum and links the medial/lateral rulers; parameter 14 uses the direct maximum.
