Complex validation through six piles
===================================

Current status: both phases pass, totaling 1,612 frames in 92 runs; 424 frames
have all six actual centroids visible. [ACCEPTANCE.md](ACCEPTANCE.md) contains
the final requirement audit and physical limits. The history below records
intermediate failures and verified milestones.

Ideal qualification passed all 898 frames in 50 runs. This
includes exact full-view counts for four, five, and six piles; mixed densities;
elongated piles; irregular layouts with additional seeds; camera entry/return;
and 184 isolated-ball control frames. There are 212 frames with all six actual
centroids visible. All matched geometry meets 0.35 m centroid / 0.30 m shape
limits, with no false piles, ID switches, or identity conflicts. Twenty-eight
C++ tests and the evaluator controls pass. Noise qualification subsequently
passed 714 frames in 42 runs on the same binaries. Proof is under `acceptance/`.

The earlier completion claim was not supported by sufficient cluster-count
coverage. The follow-up seven-pile diagnostic found real overmerging. Current
work follows the revised scope through six piles; seven-pile reports remain
historical evidence, not current acceptance runs.

First verified milestone: increasing the required density saddle from 55% to
75% of the smaller peak preserves modes separated by a 25% density valley.
This prevents distinct piles merging before persistent IDs are available;
cluster count remains inferred from density, with no fixed K or count limit.

At this milestone all 156 ideal four/five/six-pile frames and 78 ideal
six-pile mixed-density/elongated-pile frames pass. All 360 earlier regression
frames also pass. A six-pile cold-start unit regression passes, along with
the other ten tracker tests. Physical error gates remain 0.35 m centroid and
0.30 m shape, checked on every matched observation, with no count errors or
ID switches permitted.

The 132-frame wide camera sweep is still failing: 22 missing group
observations plus some geometry errors. Edge visibility and the inherited
minimum-image-pixel filter need investigation. Noise validation and the final
requirement audit are pending. The overall goal is not complete.

Evidence: `evidence/complex-baseline.json`, `complex-ideal.json`,
`complex-mixed-ideal.json`, `complex-dynamic-ideal.json`, and the three
`prominence-regression-*.json` files. Fixture files start with `complex_`.
Target artifacts are under `/root/cluster-validation/` in
`complex-baseline`, `complex-prominence`, `complex-mixed-ideal`,
`complex-dynamic-ideal`, and `prominence-regression`.

Second milestone: two density scales, verified-outline density weighting,
clipping-aware verification, and occlusion correction that does not compensate
for arcs contradicted by background pixels. Support now requires 1.5 estimated
Fuel-equivalents, retaining small piles while rejecting isolated spheres.
Contour sample count is no longer compared against image area. The 132-frame
six-pile sweep and the 184 isolated-ball controls pass; thirteen tracker tests
pass, including a distant small pile and single-ball rejection. The evaluator
also has negative controls for count, geometry, lineage, and partial detections.

Boundary-aware evaluation still requires every pile with 4+ visible balls and
uses the unchanged geometry limits. It additionally accepts predictions that
match true piles with only 1–3 visible balls, checks their location and lineage,
and rejects unmatched predictions. Strict reports remain available. This fixes
the old evaluator classifying real three-ball partial piles as false positives.

An additional irregular six-pile layout with held-out seeds exposes further
overmerging (27 missing group observations in 52 frames). A mixed-density wide
sweep also has three boundary misses in 66 frames. These failures are recorded
in `evidence/acceptance-staggered-ideal.json` and
`evidence/acceptance-mixed-dynamic-ideal.json`. Completion remains unproven;
the next step is a geometric overlap check before merging candidate modes.

Third milestone: watershed basins now accumulate weighted sample moments before
merging. A shallow smoothed valley can join untracked basins only when their
combined-covariance ellipses overlap within three sigma. Peaks inside the same
established footprint can still merge, preserving broad piles. All 52 irregular
six-pile frames and the 130 earlier ideal frames pass; fourteen tracker tests
pass. Evidence: `geometry-staggered-ideal.json` and
`geometry-regression-ideal.json`. The evaluator additionally rejects duplicate
IDs and identity reuse across different reference piles.

The mixed sweep's three remaining strict-count misses are being checked against
the actual visibility of the full cluster centroid. One frame's four ID-visible
balls appear as one overlapping RGB patch behind a field structure. An
independent field-depth centroid probe is being added; strict pixel-count
reports will remain alongside centroid-visibility checks. The goal remains open.

Fourth milestone: raw-moment overlap was too sensitive to sampling lobes in
broad piles. Candidate footprints are now compared at the existing 30 cm pile
scale using the 90% 2D Gaussian contour, with a joint-Gaussian classification
likelihood check for distant lobes of one elongated distribution. The complete
ideal matrix passes on unchanged simulated RGB. No fixed K or output count is
used by the tracker. Failed intermediate matrices remain in
`evidence/geometric-ideal-matrix.json` and `distribution-ideal-matrix.json`.

Visibility qualification is independent of tracker output: a separate one-pixel
point pass tests actual generated centroids against opaque field depth and the
frustum, with a distorted-output bounds check. All 66 mixed-sweep RGB frames are
byte-identical before/after adding probes. Three strict-count omissions involve
four-ball fragments whose full centroids are hidden; all visible-centroid recall
gates pass. Original strict count reports are retained in `acceptance/reports/`.
The tracker never reads membership labels or centroid visibility. The evaluator
now performs partial one-to-one matching with unmatched slots; its controls
reject missing visible piles, false piles, geometry errors, ID changes,
duplicate IDs, cross-pile ID reuse, and misplaced partial detections.
