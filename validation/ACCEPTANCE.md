Clustering acceptance through six piles
======================================

The deployed C++ implementation passes ideal and 10 cm positional-noise
qualification: **1,612 simulated frames in 92 runs**, including **424 frames
with all six true centroids visible**. All 832 full-view frames have exact
reference counts; 572 explicitly exercise four, five, or six piles. No false
piles, ID switches, or cross-pile identity conflicts occur. All 28 C++ tests pass.

Three ideal strict-recall omissions remain for fragments whose actual centroids
are hidden. They are retained in the reports and explained below. There are
zero visible-centroid misses and zero noisy strict-recall misses. Projection
is approximate; mathematical pixelwise perfection is not claimed.

| Matrix | Runs | Frames | Six-centroid frames | Worst centroid error | Worst shape error |
|---|---:|---:|---:|---:|---:|
| Ideal | 50 | 898 | 212 | 0.2666 m | 0.2387 m |
| Noise σ=0.10 m | 42 | 714 | 212 | 0.2962 m | 0.2759 m |

Every matched observation meets the unchanged 0.35 m centroid / 0.30 m covariance
Bures-distance limits. Bures distance measures spread and orientation in meters.
Full-view centroid residual jitter across poses is at most 6.86 cm RMS ideal and
7.12 cm RMS noisy; maximum residual ranges are 26.22 / 27.42 cm.

| Coverage | Ideal frames | Noisy frames |
|---|---:|---:|
| Compact/elongated/two/three/empty layouts | 130 | 130 |
| Initial entry/return sweeps | 100 | 100 |
| Four, five, and six piles | 156 | 156 |
| Six with unequal densities and an elongated pile | 78 | 78 |
| Irregular six-pile layout, additional seeds | 52 | 52 |
| Six-pile entry/return sweep | 132 | 132 |
| Mixed-density six-pile sweep | 66 | 66 |
| Six isolated individual balls, normal path | 52 | — |
| Six isolated individual balls, wide sweep | 132 | — |

The isolated-ball controls produce no piles. Six supplied Gaussian labels do
not force six outputs; the tracker has no fixed K or six-cluster limit. The
matrix requires exact case/seed/frame coverage. Noise is refused until ideal
qualification passes with identical C++ binary, matrix, and fixture hashes.
No implementation changed between the passing ideal and noisy phases.

Visibility and strict counts
----------------------------

The three strict-recall omissions occur in the ideal mixed sweep:
seed 101/frame 1/group 1, seed 202/frame 26/group 5, and seed 202/frame 31/group 1.
Each has four ID-visible ball fragments but a hidden full centroid. In one such
frame the RGB evidence is one overlapping patch behind a field structure.
Forcing a pile from that patch would conflict with rejecting isolated balls.

Sweep qualification requires every four-ball reference pile whose centroid is
visible. Other detected four-ball piles still undergo the same geometry and ID
checks. Predictions matching genuine piles with 1–3 visible balls require correct
location and lineage; unmatched predictions always fail. Full-view suites retain
strict counts. Original strict-count reports remain in
[acceptance/reports](acceptance/reports), including their treatment of small
partial detections. No count or geometry error is silently discarded.

`visible_cluster_centroids` comes from an independent one-pixel point render
against opaque field depth and the undistorted frustum, plus distorted output
bounds. Fuel is excluded so its own surface does not hide the semantic center;
transparent field meshes are ignored as in the per-ball ID pass. All 79 compared
RGB frames are byte-identical before/after adding probes. The tracker receives
RGB, calibration, and robot pose, never membership or visibility labels.

Requirement audit
-----------------

| Requirement in gamepiece_success_criteria.md | Authoritative evidence |
|---|---|
| Ideal 1 / Validation 2: preserve field distribution across views | Sphere-tangent projection, exact ray-cast sphere unit test, every-frame centroid/covariance comparisons, residual stability measurements, and reviewed six-pile heatmaps. The bounds above quantify “closely preserved”; exact pixelwise height recovery is not claimed. |
| Ideal 2: remain identified while centroid stays visible | Zero visible-centroid misses, switches, or identity conflicts in both matrices; field-coordinate pose compensation; empty-frame and robot-motion unit tests. |
| Ideal 3: separate entering piles without covariance inflation | 596 sweep frames, including simultaneous six-pile and mixed-density/elongated layouts; every matched shape stays bounded; weak-entrant unit test checks established spread; global ID ownership rejects reuse even without simultaneous visibility. |
| Ideal 4: semantic location, density, spread | Two-scale density watershed and physical-scale Gaussian merge checks infer count. Broad piles remain single; 184 isolated-ball frames yield no piles. Output contains weighted ground samples, location, full covariance, visible support, and support density. |
| Validation 1: simulation, then noise | Only path_camera_sim --fuel_clusters RGB. Ideal passes before --fuel_noise=0.1, with matching binary/fixture hashes. No real video or live-camera input. |
| Validation 3: count/location/shape close to truth | Exact full-view counts through six; every matched centroid ≤0.35 m and shape ≤0.30 m. All strict omissions and partial observations remain in the evidence. |
| Neutral-zone culling | Projection culls outside X 4.60–11.94 m / Y 0–8.07 m; sphere test rejects alliance-zone output. |
| Replace covariance-spike logic | Density saddles, field footprints, and Gaussian compatibility replace fixed-K/spike splitting and image-ellipse merging. |
| Separate commit worktree and frozen milestones | Implementation is on gamepiece-cluster-stability. Original ellipse-gamepiece stays at 36f2b3c with its pre-existing files. Frozen density, silhouette, validation, six-density, occlusion, geometric, ideal-six, and accepted-six snapshots preserve progress. |
| Required build/deploy | cmake --build build --parallel 6 and cmake --build build --target dev-orin succeed. Tested local/deployed hashes match; logs and 14 tracker + 9 image utility + 5 ellipse test results are under acceptance/. |

The evaluator uses maximum-cardinality one-to-one matching with unmatched slots,
so a missing pile cannot force the loss of a valid match. Controls cover eight
defect classes, both visibility policies, partial detections, and cross-pile ID
reuse without simultaneous visibility.

Reviewed artifacts and physical limits
-------------------------------------

[Six-pile camera](acceptance/six-camera.png),
[field ellipses](acceptance/six-field.png),
[heatmap sequence](acceptance/six-heatmaps.png),
[irregular layout](acceptance/staggered-field.png), and
[entering pile](acceptance/entering-field.png).
White Fuel dots are reference centers for display only. Parentheses on labels
show nearest-pile separation; the camera legend shows robot distance. The
heatmap example stays spatially coherent while total visible support varies
from 48.53 to 52.80 across thirteen poses.

The model assumes 7.5 cm radius ground-level spheres and accurate calibration
and robot poses. Hidden balls cannot supply evidence. Support/density describe
observed silhouettes, not total inventory. Touching distributions without a
resolvable density/geometric gap may form one semantic pile.

Reproduce
---------

The artifacts are on root@dev-orin:/root/cluster-validation/acceptance-final.
Ideal RGB referenced by manifests is preserved in acceptance-geometric-baseline.
Run the C++ tracker directly:

```sh
ssh root@dev-orin
cd /root
./tools/hsv_cluster_visualizer \
  --sim_run=/root/cluster-validation/acceptance-final/basic-ideal/six-7 \
  --output_dir=/root/cluster-validation/review-six
```

Generate a fresh six-pile sequence with C++:

```sh
./tools/path_camera_sim \
  --poses=/root/cluster-validation/complex_poses.json \
  --camera_config=constants/gamepiece/camera.json \
  --fuel_truth --fuel_seed=7 --fuel_noise=0 \
  --fuel_clusters='[{"center_m":[7.7,2.3],"density":100,"covariance_m2":[0.055,0.035],"rotation_rad":0.3},{"center_m":[7.7,4],"density":100,"covariance_m2":[0.065,0.035],"rotation_rad":-0.5},{"center_m":[7.7,5.7],"density":100,"covariance_m2":[0.05,0.04],"rotation_rad":0.8},{"center_m":[10.2,1],"density":100,"covariance_m2":[0.065,0.04],"rotation_rad":0.7},{"center_m":[10.2,4],"density":100,"covariance_m2":[0.055,0.035],"rotation_rad":-0.6},{"center_m":[10.2,7],"density":100,"covariance_m2":[0.05,0.035],"rotation_rad":-0.2}]' \
  --output_dir=/root/cluster-validation/manual-six
./tools/hsv_cluster_visualizer \
  --sim_run=/root/cluster-validation/manual-six \
  --output_dir=/root/cluster-validation/manual-six/clusters
```

For the full matrix, from the implementation worktree:

```sh
rsync -a scripts/*gamepiece*.py validation/*.json root@dev-orin:/root/cluster-validation/
ssh root@dev-orin 'cd /root && python3 cluster-validation/gamepiece_acceptance.py run --phase ideal --root /root/cluster-validation/reproduce-six'
rsync -r --include='*/' --include='manifest.json' --include='clusters.json' --include='*-phase.json' --exclude='*' root@dev-orin:/root/cluster-validation/reproduce-six/ /tmp/reproduce-six/
python3 scripts/gamepiece_acceptance.py check --phase ideal --root /tmp/reproduce-six --fixtures validation
rsync /tmp/reproduce-six/ideal-report.json root@dev-orin:/root/cluster-validation/reproduce-six/
ssh root@dev-orin 'cd /root && python3 cluster-validation/gamepiece_acceptance.py run --phase noisy --root /root/cluster-validation/reproduce-six'
rsync -r --include='*/' --include='manifest.json' --include='clusters.json' --include='*-phase.json' --exclude='*' root@dev-orin:/root/cluster-validation/reproduce-six/ /tmp/reproduce-six/
python3 scripts/gamepiece_acceptance.py check --phase noisy --root /tmp/reproduce-six --fixtures validation
```

The local checker needs NumPy/SciPy; the target runner uses the standard library.
Unchanged simulated RGB can be reused with run --replay-from OLD_ROOT.
Summaries, per-frame metrics, visibility checks, hashes, tests, build/deploy logs,
and stability measurements are committed under acceptance/.
