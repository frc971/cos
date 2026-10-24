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
pose for stationary-camera robot coordinates. The callback reads the latest
WPILib `Pose3d` from the NetworkTables topic `/Robot/Pose` once per frame.
Configure the constructor's `pose_topic` to match the Java pose provider; an
empty topic disables the subscription. Before a pose is available, processing
uses robot coordinates; unavailable or nonfinite samples retain the last pose.
For replay, an explicit constructor `pose_path` instead consumes a context
`localization::PositionEstimateMessage`; a missing estimate retains the last pose. Camera calibration and pose must be
accurate. Switching between field and robot coordinates resets the ID map.

Each result retains normalized image samples and robot-relative ground samples,
weights, centroid, covariance, persistent ID, and visible support. `density` is
the canonical per-pile scalar: estimated visible ball count divided by effective
Gaussian footprint area. Verified sphere arcs anchor the count; unexplained
yellow area and internal image edges supply a calibrated correction for partial
balls. Calibration excludes balls with zero rendered image evidence. The
estimated/verified counts, residual evidence, and `density_area_m2` are included
in C++ results for inspection. See
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

The hardware integration test replays three grayscale localization cameras and
one color gamepiece camera concurrently. Both real and simulated recordings
must contain JPEGs named by their capture timestamps in seconds on the same
clock. Localization runs only to apply load; its output never feeds the tracker.
The tracker instead receives interpolated reference poses from a JSON array:

```json
[
  {"timestamp": 100.0, "x": 2.0, "y": 4.0, "yaw": 0.0},
  {"timestamp": 105.0, "x": 2.5, "y": 4.0, "yaw": 0.1}
]
```

Translation is in WPILib field meters and yaw in radians. Samples must be finite,
strictly increasing, and cover the entire color recording. Translation is
interpolated linearly and yaw follows the shortest angular path. All four
recordings must overlap. No simulator manifest or simulator-specific input mode
is supported; prepare the same folders, calibrations, and pose file for any source.

Run on dev-orin from `/root`, replacing paths with the recording to test:

```sh
./tests/hsv_cluster_pipeline_test \
  --localization_logs=/logs/gray/front,/logs/gray/left,/logs/gray/right \
  --localization_configs=constants/second_bot/front_camera.json,constants/second_bot/left_camera.json,constants/second_bot/right_camera.json \
  --gamepiece_log=/logs/color --gamepiece_config=constants/gamepiece/camera.json \
  --poses=/logs/poses.json --output_dir=/root/debug-output/replay
```

The output directory must be new or empty. The test writes only two images:
`overall_heatmap.png` accumulates weighted ground support, and
`field_overlay.png` shows the robot path and one median-area observed ellipse per
persistent cluster ID. `summary.json` records frame counts, skipped frames,
localization pose counts, tracker FPS, and capacity/throughput for each camera.

Add `--ground_truth=/logs/clusters.json` for accuracy metrics. This optional file
uses absolute JPEG timestamps and independently annotated field centers/densities:

```json
[{"timestamp":100.0,"clusters":[
  {"x":6.0,"y":4.0,"density":20.0,"in_frame_fraction":1.0}
]}]
```

Supply every color timestamp (within 1 µs), including empty cluster arrays.
`in_frame_fraction` is the fraction of a cluster's gamepieces on-screen; it
defaults to 1. Clusters with at least two-thirds off-screen are excluded.
`summary.json.accuracy` reports missed cluster observations, median and population
standard deviation of matched Euclidean centroid errors in meters, and incorrect
density-pair orderings with their denominator and error fraction. Matching is
one-to-one within 0.9 m, maximizing matches before minimizing distance. Density
comparisons use matched pairs in the same frame; equal true densities are ignored
and predicted ties count as incorrect. No matches/pairs produce null error metrics.
Accuracy covers processed color frames over the full recording; skipped frames
remain in the camera statistics. These metrics do not add acceptance thresholds.

Each grayscale stream and the color pipeline must exceed **30 FPS processing
capacity** during their common recording interval. Capacity is completed frames
divided by summed dispatch-to-completion times, including decoding, worker queue
waits, and downstream processing. Replay pacing and final rendering are excluded.
Grayscale completion includes detection and fused localization, even when no tags
or poses are found. Color completion includes decoding and tracking; tracker-only
FPS starts at decode completion. Wall FPS includes pacing over the entire replay
and is reported separately. At least two frames per stream must be measured, every
dispatched frame must finish, and at least one valid cluster must be observed.
All streams drain independently; the first EOF does not end the test.
`--timeout_seconds` defaults to 300. A failed performance gate still writes results.

Run the compact edge-case tests and the input/accuracy checks separately:

```sh
./tests/hsv_cluster_tracker_node_test
./tests/hsv_cluster_pipeline_test --self_test
```

The integration replay is excluded from automatic CTest runs. Its input and
accuracy checks are registered as `hsv_cluster_pipeline_input_test`.
