The tracker thresholds yellow Fuel (hue 18–30, saturation ≥100) and projects
silhouette tangents of 7.5 cm radius spheres onto the field. Pile geometry uses
contour evidence; a separate density pass verifies individual sphere arcs and
measures residual image evidence. Outline consistency rejects unreliable geometry
and estimates visible perimeter for occlusion weighting. All tangent evidence
is retained at reduced weight for finding piles whose full outlines are not
yet visible. Verification considers the part of an outline inside the image;
occlusion correction excludes arcs contradicted by non-yellow pixels.

Clustering uses a 10 cm grid and density saddles at two scales: 30 cm resolves
weak entering peaks, while 40 cm merges sampling bumps along broad piles.
Untracked modes remain separate across a density valley of at least 25%.
Before merging untracked modes, their centers must fall within the 90% contour
of their combined covariance at the pile smoothing scale, or one Gaussian must
explain their samples better than two classified Gaussians. This preserves
compact separated piles and joins sampling lobes of elongated piles. Modes
inside the same tracked footprint can merge.
Established field centroids and covariance footprints keep a separate entering
mode from inflating an existing pile. There is no fixed cluster count, k-means
loss-driven splitting, covariance-spike trigger, or image-ellipse merging.
A minimum 1.5 Fuel-equivalents of raw or verified, occlusion-corrected support
suppresses isolated balls. There is no image-area cutoff on contour samples.

`ProcessFrame(frame, robot_pose)` uses field coordinates for clustering, neutral
zone culling (X 4.60–11.94 m, Y 0–8.07 m), and one-to-one ID matching. Omit the
pose for stationary-camera robot coordinates. The callback accepts an optional
constructor `pose_path` containing `localization::PositionEstimateMessage`;
a missing estimate holds the last pose. Camera calibration and pose must be
accurate. Switching between field and robot coordinates resets the ID map.

Each result retains normalized image samples and robot-relative ground samples,
weights, centroid, covariance, persistent ID, and visible support. `density` is
the canonical per-pile scalar: estimated visible ball count divided by effective
Gaussian footprint area. Verified sphere arcs anchor the count; unexplained
yellow area and internal image edges supply a calibrated correction for partial
balls. Calibration excludes balls with zero rendered image evidence. The
estimated/verified counts, residual evidence, and `density_area_m2` are included
in C++ results and visualizer JSON for inspection. See
[initial density validation](../../validation/density/RESULTS.md). The default
density model predicts stationary field-space sphere hypotheses before searching
unexplained edges for new balls. Count and covariance estimates persist between
frames; current image evidence is still required. See
[temporal density validation](../../validation/temporal-density/RESULTS.md).
IDs survive empty
frames and re-entry. Ground covariance and
centroid are computed from projected samples, rather than projecting an image
mean or covariance through a nonlinear transform.

Pose-noise and blind-tracking measurements are in the
[pose sensitivity report](../../validation/pose-sensitivity/RESULTS.md), including
relative geometry, field error, ID changes, and density stability.

Build and deploy:

```sh
cmake --build build --parallel 6
cmake --build build --target dev-orin
```

Validate exclusively with simulated Gaussian layouts. Copy the validation runner
and scenarios to the target, then run ideal cases before positional noise:

```sh
rsync -a scripts/run_gamepiece_validation.py validation/scenarios.json root@dev-orin:/root/cluster-validation/
ssh root@dev-orin 'cd /root && python3 /root/cluster-validation/run_gamepiece_validation.py --root=/root/cluster-validation/ideal-suite'
ssh root@dev-orin 'cd /root && python3 /root/cluster-validation/run_gamepiece_validation.py --root=/root/cluster-validation/noisy-suite --noise=0.1'
ssh root@dev-orin 'cd /root && python3 /root/cluster-validation/run_gamepiece_validation.py --root=/root/cluster-validation/dynamic-suite --dynamic --only=two,elongated'
```

The simulator's `--fuel_truth` records cluster membership and an additional
occlusion-aware ID render's per-ball visible pixel counts. Only the evaluator
reads these IDs; the tracker consumes RGB and calibration/robot pose. White dots
in simulated field views show generated centers for visual comparison.

Fetch each suite's manifests and `clusters/clusters.json` (preserving folders),
then run `scripts/evaluate_gamepiece_validation.py SUITE --output=REPORT.json`
with Python NumPy, SciPy, and OpenCV. It checks every frame's count, one-to-one lineage,
centroid, and covariance against visible generated centers. Fixed gates are no
misses/extras/ID switches, every matched centroid error ≤0.35 m, and every
covariance Bures shape distance ≤0.30 m (p95 values are also reported). The latter measures ellipse spread and orientation in
meters; it cannot be passed just by matching centroids. Samples with fewer than
four visible balls are below the evaluator's pile visibility threshold. Empty
runs must yield no clusters. `--allow-partial` also permits predictions matching
true piles with 1–3 visible balls, checks their location and IDs, and still
requires every pile with at least four visible balls, except when at least 2/3
of the pile's generated balls are off-screen. The coverage test uses projected
sphere silhouettes, counts partial silhouettes as on-screen, and ignores
occlusion; a hidden ball inside the sensor still counts toward coverage.
An exempt pile remains available for matching and geometry measurements but
cannot contribute a miss. The same rule applies to pose-sensitivity reports.
`ignored_offscreen_observations` records exempt pile/frame pairs and
`ignored_offscreen_misses` records the misses actually suppressed. Evaluation
needs `gamepieces_in_frame` flags or render calibration and reference poses;
`render_camera_calibration` preserves true rendering calibration when replay
calibration differs. These fields never enter the tracker. Strict count reports are
retained alongside this boundary-aware report. Negative evaluator controls
check missed piles, false piles, geometry errors, ID changes, and misplaced
partial detections: `python3 scripts/test_gamepiece_evaluator.py`.

`--centroid-gate` uses a separate simulator field-depth point pass to require
recall while the true cluster centroid is visible. It still records every
nonexempt four-ball reference miss in `misses` and `missed_observations`, and separately
reports `centroid_visible_misses` and `hidden_centroid_misses`. Geometry, false
predictions, duplicate IDs, cross-pile ID reuse, and lineage gates always apply.
This handles field-occluded fragments without forcing a pile detection from an
RGB patch indistinguishable from one ball. Full-view suites use strict counts.

Validation must graduate through four, five, and six piles. The `complex_*`,
`staggered_scenarios.json`, and `speck_scenarios.json` fixtures include unequal
densities, an elongated pile, irregular layouts, camera entry/return, and
isolated-ball controls. See `validation/COMPLEX_PROGRESS.md` for current evidence
and the status of the full goal. Historical seven-pile reports describe the
earlier failure; current acceptance scenarios max out at six.

The reproducible matrix is `validation/acceptance_suites.json`. Run
`scripts/gamepiece_acceptance.py run --phase ideal --root ROOT` on dev-orin
from `/root`, after copying the scripts and top-level validation JSON files into
`/root/cluster-validation`. Download the resulting manifests, cluster JSON, and
`ideal-phase.json`; locally run the same script with `check --phase ideal
--root DOWNLOADED_ROOT --fixtures validation`. Copy its `ideal-report.json` back
to ROOT, then run the noisy phase with the same binaries. The runner refuses
noise until the ideal report passes and verifies binary/fixture hashes. Checks
require complete case/seed/frame coverage, four/five/six observed groups, and
actual frames with all six centroids visible. Per-suite original strict reports
remain alongside the centroid-gated camera sweeps.

Visualize any simulator run directly:

```sh
ssh root@dev-orin 'cd /root && ./tools/hsv_cluster_visualizer --sim_run=/root/cluster-validation/acceptance-final/basic-ideal/six-7 --output_dir=/root/cluster-validation/review-six'
```

Outputs include camera/field views, per-frame and overall weighted heatmaps,
comparison panels, and JSON with ground covariance, visible support, ball count
evidence, and `density`. Colored rings in field views mark verified spheres.
The cluster tracker and camera annotations use the input pixels directly with
pinhole intrinsics; the tracker has no distortion parameter or rectification stage.
The simulator can still render lens distortion for validation. AprilTag localization
retains its own calibration. Frame processing uses exact simulated poses, never
inferred ground-truth ball positions.
