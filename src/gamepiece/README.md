The tracker thresholds yellow Fuel (hue 18–30, saturation ≥100) and projects
silhouette tangents of 7.5 cm radius spheres onto the field. It does not detect
or identify individual balls. Outline consistency rejects unreliable geometry
and estimates visible perimeter for occlusion weighting. All tangent evidence
is retained for finding piles, including newly entering piles whose full outlines
are not yet visible.

Clustering uses a 10 cm grid, 40 cm Gaussian bandwidth, and density saddles.
Established field centroids and covariance footprints keep a separate entering
mode from inflating an existing pile. There is no fixed cluster count, k-means
loss-driven splitting, covariance-spike trigger, or image-ellipse merging.
A minimum 2.5 Fuel-equivalents of raw silhouette support suppresses isolated
specks; this is a pile detector, not an individual-ball detector.

`ProcessFrame(frame, robot_pose)` uses field coordinates for clustering, neutral
zone culling (X 4.60–11.94 m, Y 0–8.07 m), and one-to-one ID matching. Omit the
pose for stationary-camera robot coordinates. The callback accepts an optional
constructor `pose_path` containing `localization::PositionEstimateMessage`;
a missing estimate holds the last pose. Camera calibration and pose must be
accurate. Switching between field and robot coordinates resets the ID map.

Each result retains normalized image samples and robot-relative ground samples,
weights, centroid, covariance, persistent ID, and visible support. The support
and derived peak density describe visible evidence, not a complete inventory of
occluded balls. IDs survive empty frames and re-entry. Ground covariance and
centroid are computed from projected samples, rather than projecting an image
mean or covariance through a nonlinear transform.

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
ssh root@dev-orin 'cd /root && python3 /root/cluster-validation/run_gamepiece_validation.py --root=/root/cluster-validation/dynamic-suite --dynamic'
```

The simulator's `--fuel_truth` records cluster membership and an additional
occlusion-aware ID render's per-ball visible pixel counts. Only the evaluator
reads these IDs; the tracker consumes RGB and calibration/robot pose. White dots
in simulated field views show generated centers for visual comparison.

Fetch each suite's manifests and `clusters/clusters.json` (preserving folders),
then run `scripts/evaluate_gamepiece_validation.py SUITE --output=REPORT.json`
with Python NumPy and SciPy. It checks every frame's count, one-to-one lineage,
centroid, and covariance against visible generated centers. Fixed gates are no
misses/extras/ID switches, every matched centroid error ≤0.35 m, and every
covariance Bures shape distance ≤0.30 m (p95 values are also reported). The latter measures ellipse spread and orientation in
meters; it cannot be passed just by matching centroids. Samples with fewer than
four visible balls are below the evaluator's pile visibility threshold. Empty
runs must yield no clusters. See `validation/PROGRESS.md` for measured results.

Visualize any simulator run directly:

```sh
ssh root@dev-orin 'cd /root && ./tools/hsv_cluster_visualizer --sim_run=/root/cluster-validation/ideal-suite/two-7 --output_dir=/root/cluster-validation/review'
```

Outputs include camera/field views, per-frame and overall weighted heatmaps,
comparison panels, and JSON with ground covariance, visible support, and peak
density. Lens distortion follows the manifest's calibration settings. Frame
processing uses exact simulated poses, never inferred ground-truth ball positions.
