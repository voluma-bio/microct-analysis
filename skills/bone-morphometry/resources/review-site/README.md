# Local micro-CT review site

Interactive review of the OA6-1RK analysis: 12 individually selectable structures, physical-coordinate rulers, raw/filtered/aligned image stacks, registered slice planes, evidence history, and browser-local review notes. No analysis data is modified by the site.

The exporter targets the documented `data/python-protocol-run` layout. It uses only explicitly named review directories, never supplemental tables. Generated assets stay under ignored `data/`. Meshes are the existing step-2 review surfaces, without added smoothing or decimation. Calculations remain in original image coordinates. Cropped bone label volumes are not whole-bone volumes.

From the repository root:

```bash
npm ci --prefix skills/bone-morphometry/resources/review-site --ignore-scripts
MPLCONFIGDIR=/tmp/microct-mpl .venv/bin/python skills/bone-morphometry/resources/review-site/export.py data/python-protocol-run data/python-protocol-run/web-review
.venv/bin/python skills/bone-morphometry/resources/review-site/server.py --run data/python-protocol-run --assets data/python-protocol-run/web-review --port 8765
```

Open `http://127.0.0.1:8765`. Dependencies are served locally, with no runtime CDN requests. The server reads memory-mapped volumes and generates PNG slices on request. Only the UI, installed Three.js assets, and exported review files are served; there is no arbitrary filesystem or write API.

For private Tailscale access, first inspect `tailscale serve status` to avoid replacing another service. With HTTPS port 8443 unused:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8765
tailscale serve status
```

This uses Serve, not public Funnel. Access follows the tailnet's access policy. The Python process must remain running; the Serve configuration alone does not restart Python. Stop only this route with `tailscale serve --https=8443 off`.

## Reviewing

1. Select a measurement, then focus it or isolate its associated bone. Use the anatomical femoral cameras to inspect A and C from the correct sides.
2. Surfaces are fully opaque by default. Rulers and labels are occluded by visible bones; turn on Show hidden rulers only when projection through bone is wanted. The surface is sampled at two voxels, so small endpoint/surface offsets are expected.
3. Open endpoint slices, or scroll through all three source-grid planes and the tibia-aligned volume. Toggle segmentation to inspect the HU image beneath it. Add a slice plane to the 3D scene to check registration.
4. Use each structure's Review button to record inclusion or boundary concerns. Femur, tibia and fibula volumes are limited by scan coverage.
5. Record a decision and notes, then export them as JSON. They are saved in this browser only, not shared across devices and not automatically applied to landmarks. The export retains grid identity and measurement coordinates. Saved 3D PNGs include ruler labels; evidence thumbnails open at full size.
6. Compare earlier segmentation rounds in the evidence selector. Review the report and coordinate/provenance downloads before accepting a result.

A provisional acceptance is a review note, not validation of the entire study protocol. Tibial maximum-height landmarks, medial growth plate, paper-defined tibial width, and subchondral/trabecular ROIs remain unresolved.

## Verification

`tests/test_review_site.py` checks slice axis orientation, overlay preservation, and rejection of invalid requests. The exporter asserts that transformed scene endpoints reproduce original physical distances. Live browser checks cover all 12 meshes, structure isolation during loading, registered slice display, note persistence, evidence links, and screenshots. Python lint and JS syntax are checked separately.

## Current OA6-1RK deployment

Private URL: https://pop-os.tail852a76.ts.net:8443/ . The user-session service
`microct-review.service` runs the server on loopback port 8766. Inspect with
`systemctl --user status microct-review.service`; restart with
`systemctl --user restart microct-review.service`. This is a transient service,
so it is not installed for automatic recreation after a reboot. Tailscale Serve
persists separately. To recreate the service after reboot:

```bash
systemd-run --user --unit=microct-review --description='OA6-1RK private micro-CT review' --property=Restart=on-failure --property=WorkingDirectory=/home/jimyao/gitrepos/prompts/microct-analysis /home/jimyao/gitrepos/prompts/microct-analysis/.venv/bin/python /home/jimyao/gitrepos/prompts/microct-analysis/skills/bone-morphometry/resources/review-site/server.py --run /home/jimyao/gitrepos/prompts/microct-analysis/data/python-protocol-run --assets /home/jimyao/gitrepos/prompts/microct-analysis/data/python-protocol-run/web-review --port 8766
```

The exporter accepts `--study-pdf PATH` to include only the explicitly supplied
main manuscript. Source scripts and intake/filter provenance are also exported.

## Presentation parameter inventory

`parameters.json` records all 23 numbered definitions from the supplied
`uCT analysis parameters.pptx`. `parameter-panel.js` connects each definition to
its source slide, related measurements and candidate anatomical labels. The
presentation PDF, slide images, copied PPTX and source hash are stored under
`data/python-protocol-run/parameter-deck/` and exported as review evidence.

The inventory is not a claim that all parameters have been measured. Prior
estimates retain their original provenance. Candidate joint-mineralization
volumes are not accepted meniscus volumes. Unresolved parameters have no value
or ruler. Slide 5's vertebral parameters 15–23 are listed as unavailable because
this dataset has no vertebra. Slide 6 is blank; slide 7 adds illustrative images.

The deck specifies femoral width using a lateral ridge midpoint and medial
vertex, so the earlier W=0 extrema measurement is explicitly marked as a
definition mismatch. Tibial trabecular ROI offsets are recorded as 50 µm distal,
200 µm long and 20 µm inward from cortex; no ROI mask is invented before the
growth-plate and cortical boundaries are accepted. Cortical thickness locations
at 100 versus 500 µm remain unresolved candidates.

Run the parameter-mapping checks with:

```bash
node --test skills/bone-morphometry/resources/review-site/parameter-panel.test.js
```

The exporter now includes the supplemental-comparison and placement-audit
directories, so rebuilding review assets retains those investigation records.

Slide 4 also contributes an unnumbered exploratory node-to-end ratio. Slide 2
illustrates a slanted growth-plate-level width; no common transverse plane is
imposed by the catalog. Previous tibial endpoints still require validation.
