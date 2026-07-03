# Subchondral Bone / OA Mouse Knee Cues

For a vague prompt like “mouse knee osteoarthritis study; understand
subchondral bone changes,” a strong design should consider:

- tibial and/or femoral subchondral compartments, depending on scan coverage
- medial vs lateral compartment comparison, especially in OA models with
  compartment-specific loading
- depth-bounded VOIs from the articular surface rather than whole-bone averages
- separating subchondral plate from trabecular epiphysis when image quality allows
- BV/TV and trabecular metrics inside the VOI, plus plate/cortical thickness when
  the research question concerns sclerosis
- explicit exclusion or handling of growth plate, osteophytes, and cortical shell
- QC overlays showing articular surface, VOI depth, medial/lateral split, and
  thresholded bone mask

Failure patterns:

- computing whole tibia/femur metrics for a subchondral question
- ignoring medial/lateral asymmetry
- reporting BV/TV without tissue-volume denominator and ROI definition
- using pixel counts without voxel spacing
