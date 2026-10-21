# Python reference workflow for the OA paper

This implements calibrated DICOM intake, explicit 3D median filtering, seeded
watershed segmentation, anatomical reslicing, and annotation-driven geometry,
label volumes, and regional BV/TV. It produces 3D screenshots and slice overlays
for inspection. It does not automatically identify anatomical landmarks or
reproduce Amira/Scanco internals exactly.

The source is the main manuscript *Geometric indices derived from µCT images as
sensitive and reliable parameters for evaluating disease severity and therapeutic
response of osteoarthritis in mice*, methods pages 5–9 and figure legends.
Supplemental tables and historical expected measurements are not inputs.

## Calibration

Apply DICOM rescaling once to the stored signed pixels. The local OA6-1RK series
has the same mapping across all 877 slices:

```text
HU = 0.391063 * stored_value - 1000
linear_attenuation_per_cm = stored_value / 4096
HU = 1000 * (linear_attenuation_per_cm / 0.62430 - 1)
```

The latter two equations independently agree with the public rescale tags using
the private Scanco calibration fields. The absence of `RescaleType` is permitted
for this ORIGINAL, non-LOCALIZER, non-multi-energy CT series: its output is HU.
This establishes the encoded mapping, not independent phantom accuracy. Do not
force the solvent/soft-tissue histogram peak to zero or call Otsu calibration.

| Paper branch | Stated threshold | This sample's corresponding HU |
|---|---|---|
| Amira bone mask | >2500 HU | >2500 |
| Amira seed interval | 3000–5000 HU | 3000–5000 |
| Scanco subchondral plate/cortical definition | 270 = 2.16 cm⁻¹ | 2459.875060 |
| Scanco general 3D image | 320 = 2.56 cm⁻¹ | 3100.592664 |

The paper also reports a general Scanco setting of 220; this implementation
does not silently substitute that setting for either analysis branch. Explicitly
record the branch and threshold when annotating a regional measurement.

References: [DICOM CT Image Module](https://dicom.nema.org/medical/dicom/2025d/output/chtml/part03/sect_C.8.2.html)
and [vtk-dicom Scanco conversion documentation](https://dgobbi.github.io/vtk-dicom/doc/api/classvtkScancoCTReader.html).

## Commands

Run from the package root using its existing `.venv`. Each output directory must
be new/empty, preserving earlier inspection rounds. Large data stay under the
Git-ignored `data/` directory. `MPLCONFIGDIR=/tmp/microct-mpl` avoids a read-only
Matplotlib home cache in restricted environments.

```bash
.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py intake \
  data/OA6-1RK data/my-run/intake

.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py preview \
  data/my-run/intake data/my-run/review-01

# 3 is an example explicit choice, not a kernel specified by the paper.
.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py prepare \
  data/my-run/intake data/my-run/prepared --median-size 3

.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py segment \
  data/my-run/prepared data/my-run/review-02 --seeds data/my-run/seeds.json

.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py reslice \
  data/my-run/intake data/my-run/aligned --axes data/my-run/axes.json

.venv/bin/python skills/bone-morphometry/resources/python-protocol/workflow.py measure \
  data/my-run/prepared data/my-run/measurements --spec data/my-run/measurements.json \
  --labels data/my-run/review-02/labels.npy
```

`preview` uses connected components solely to inspect the threshold result. Its
numeric labels are not bone names. `--display-labels 1 2 ...` limits the 3D view;
saved label arrays and slice overlays retain all components above the explicitly
recorded display-size cutoff. That cutoff is not used by `segment` or `measure`.

`segment` uses gradient-magnitude watershed with six-neighbor connectivity,
inside the >2500 HU mask. Those algorithm details are implementation choices:
the paper does not specify Amira's elevation/connectivity settings. Unseeded
disconnected objects remain label 0; their voxel count is reported. The marker
click expands through its connected 3000–5000 HU region, similar to a 3D magic
wand. If two bones share that region, supply reviewed `--markers markers.npy`
instead of changing thresholds to force a separation.

## Annotation contracts

Every annotation document includes `grid_id` from its originating scan metadata.
Arrays are Z,Y,X. Distance points are X,Y,Z voxel-center indices. The affine maps
XYZ indices into physical millimeters. Seeds use integer Z,Y,X indices:

```json
{
  "grid_id": "COPY_FROM_ORIGINATING_SCAN",
  "structures": [
    {"label": 1, "name": "femur", "points_zyx": []},
    {"label": 2, "name": "tibia", "points_zyx": []}
  ]
}
```

Fill points only after inspecting the actual anatomy. Empty points intentionally
fail; the template is not a runnable scientific annotation. Explicit marker
arrays require positive integer structure IDs and zeros elsewhere; selected
voxels must remain inside the paper's seed interval.

`axes.json` is a right-handed orthonormal 3×3 matrix whose columns are the desired
anatomical X,Y,Z axes in DICOM world coordinates. Identity is scanner orientation,
not an assertion of anatomical alignment. Reslicing uses linear interpolation,
preserves the field of view, records its new affine, and changes the grid identity.
Existing masks must be re-annotated or properly resampled with nearest-neighbor
interpolation and reviewed before binding them to that new grid.

Measurement JSON supports:

- `distances`: named `{points_xyz: [[x,y,z], [x,y,z]], definition: "..."}` entries.
- `slice_heights`: named `{axis_xyz: 2, start_index: N, end_index: M,
  orientation_reviewed: true, definition: "..."}` entries.
- `ratios`: named `{numerator: "distance_name", denominator: "distance_name"}` entries.
- `label_volumes`: names mapped to integer segmentation IDs.
- `regions`: named `{mask_npy: "/absolute/path/roi.npy", threshold_hu: number,
  definition: "boundary and included/excluded tissues"}` entries. ROI masks must
  be nonempty boolean arrays representing the full tissue region, including
  marrow; BV/TV is bone voxels inside that region divided by its total voxels.

External marker, label, and ROI arrays require a same-stem JSON sidecar containing
their `grid_id` and SHA256. Create it with `save_bound_array(path, array, meta)`
from `workflow.py` at annotation time. Never attach a new grid ID to an old mask
merely to pass validation. Output measurements retain mask hashes, and command
provenance retains label hashes and the full supplied specification.

## Anatomical definitions and review

- Femoral length: upper midpoint of intercondylar groove to intercondylar notch.
- Femoral width: medial to lateral condylar edges. Preserve true osteophytes.
- Femoral index: width / length.
- Tibial maximum IIOC height: articular-surface appearance to growth-plate
  appearance on reviewed anatomically aligned slices. This implementation uses
  slice-center separation (`abs(end-start) * spacing`), not an inclusive count.
  The manuscript does not specify the endpoint counting convention.
- Tibial width: explicitly record the selected level. The methods describe
  condylar borders; Figure 3 specifies the growth-plate level.
- Medial/lateral heights: plateau midpoint to epiphyseal line on the selected
  frontal slice. Record the slice and anatomical side.
- Tibial indices: each selected height / tibial width.
- Patella/meniscus label volumes reflect threshold-visible mineralized tissue.
  Do not interpret them as complete soft-tissue meniscal volumes.

For every meaningful change, inspect at least opposing 3D views and slice
overlays, record the issue and correction, and render again into a fresh output
directory. Camera poses and meshes are saved. Display meshes use a two-voxel
marching-cubes step and no smoothing; padded boundary caps are display artifacts.
They are not used for numerical distances or volume calculations. Check full
resolution slices when a thin connection or landmark is uncertain.

Do not accept measurements while femur/tibia are fused, the growth plate is
ambiguous, the selected ROI excludes marrow unintentionally, or landmarks are
unreviewed. Grid/hash validation proves input identity, not anatomical accuracy.

## Scope and verification

The standalone CLI implements the geometric branch and BV/TV bookkeeping. It
does not implement validated Scanco-equivalent Tb.Th, Tb.Sp, or Tb.N, histology,
or group statistics. The old runtime's mean distance-transform helpers are not
substitutes for a validated trabecular morphometry implementation.

Run `MPLCONFIGDIR=/tmp/microct-mpl .venv/bin/python -m pytest tests/test_paper_protocol.py -q`.
Tests cover calibration disagreement, coordinate geometry, slice intervals,
ROI denominators, wrong-grid/modified-mask rejection, connected seed collisions,
and identity reslicing. These numerical tests do not establish biological accuracy.
