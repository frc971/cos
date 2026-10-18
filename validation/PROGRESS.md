Current six-pile acceptance passes 1,612 frames across 92 ideal/noisy runs,
including 424 frames with all six true centroids visible. See
[ACCEPTANCE.md](ACCEPTANCE.md) for current results, visibility limits, the
requirement audit, images, and C++ reproduction commands. The results below describe
the earlier, smaller layouts only. [SEVEN_CLUSTERS.md](SEVEN_CLUSTERS.md) records
the historical seven-pile failure at 7871743; current acceptance maxes out at six.

Completed initial validation on root@dev-orin, 2026-10-18. All tests used simulated
RGB, exact camera/robot poses, and separately recorded ground truth. No real
video or live-camera inputs were used.

| Suite | Runs | Frames | Worst centroid error (m) | Worst shape Bures error (m) |
|---|---:|---:|---:|---:|
| ideal | 10 | 130 | 0.2110 | 0.2128 |
| noisy | 10 | 130 | 0.3051 | 0.2442 |
| dynamic | 4 | 100 | 0.3413 | 0.2257 |

Across all 360 frames: zero missed groups, zero extra groups, zero ID switches.
Every matched centroid is within 0.35 m and every covariance shape distance is
within 0.30 m. The evaluator also records p95 errors and per-frame match metrics.
The geometric requirements are checked on every match, not only an average.
These numerical bounds quantify "close"; this is not a claim of mathematically
perfect projection or knowledge of completely occluded balls.

The ideal suite covers compact and elongated single piles, two anisotropic
piles, three piles, and an empty field, each with seeds 7 and 23 over 13 moving
camera poses. After those passed, the same suite graduated to --fuel_noise=0.1.
The 25-pose dynamic sweep tests elongated partial views, new piles entering,
centroids leaving view, and return to earlier views. Both dynamic scenarios use
two seeds. Generated ball centers with at least eight visible ID-render pixels
form the reference distribution; four visible balls establish a reference pile.
The tracker never sees membership IDs, visible counts, or generated positions.

The committed scenarios and scripts reproduce the tests. Full target artifacts:
root@dev-orin:/root/cluster-validation/{ideal-suite,noisy-suite,dynamic-suite}.
Their clusters folders contain camera views, ground ellipses, heatmaps, and JSON.
The local evidence/*.json reports contain all frame comparisons. White dots in
simulation field views show generated centers only for display.

Requirement-by-requirement completion evidence:

| Requirement | Current evidence |
|---|---|
| Preserve projected distribution shape | Exact ray-cast sphere projection test; all 360 covariance comparisons; reviewed field and heatmap PNGs. Tangent geometry uses the sphere center-height constraint, not the lowest ball in an entire column. |
| Stable visible centroids and IDs | No count misses or switches in any suite, including entry/return; unit tests retain IDs through empty frames and compensate robot translation/rotation. |
| Identify genuinely new piles | Dynamic two-pile sequences preserve separate IDs; weak entering-mode test bounds the established pile's covariance. Prior field footprints protect new modes outside their geometry. |
| Semantic location, density, spread | Broad piles stay single and separated piles remain distinct; results include robot-relative centroids, full ground covariance, visible-support weights, and peak density. No k-means loss or fixed K determines output count. |
| Simulated data before noise | Ideal gate passed before noisy-suite creation; runner uses --fuel_clusters and subsequently --fuel_noise=0.1, with recorded seeds and per-ball truth. |
| Number/location/shape close to truth | All three evidence reports pass exact counts, lineage, per-match centroid ≤0.35 m, and per-match shape distance ≤0.30 m. Empty scenes also pass. |
| Neutral-zone culling | Projection culls outside neutral X/Y bounds when pose is available; sphere test moves the identical pile into the alliance zone and verifies no output. |
| Concrete geometry replaces covariance spike | Density saddles, physical smoothing bandwidth, and prior 2-sigma footprints replace fixed-K/spike splitting and image-ellipse merging. |
| Separate commit worktree and frozen milestones | Implementation branch is gamepiece-cluster-stability; original ellipse-gamepiece HEAD remains 36f2b3c. Frozen density and silhouette snapshots are 19d2ca9 and 7545dab; the final frozen worktree records the completed implementation. |
| Required build/deploy | cmake --build build --parallel 6 and cmake --build build --target dev-orin both exit 0. Tracker 10/10, image utilities 9/9, ellipse 5/5 tests pass on dev-orin. |

The image utility GPU tests need the matching OpenCV CUDA shared libraries that
are absent from the target image. They were copied from the build sysroot into
/root/cluster-validation/test-libs and used through LD_LIBRARY_PATH, leaving
system libraries untouched. The tracker and visualizer run using normal target
libraries. The callback regression test covers padded GPU BGR input and the
optional localization pose message.

Worktrees are frozen at significant verified milestones, rather than at failed
experiments. The first density checkpoint corrected counts/lineage on the initial
17-frame approach; the silhouette checkpoint passed 130 ideal frames. The final
checkpoint adds outline verification/occlusion weighting, geometric protection
for entering modes, production callback pose input, and all 360 passing frames.

Known physical limits: ball radius is fixed at 0.075 m; calibration/robot pose
must be correct; fully hidden balls cannot contribute evidence; groups with fewer
than about 2.5 raw Fuel-equivalents are filtered; touching distributions without
a density/geometry separation can form one pile. Support/density is an estimate
of observed evidence. None of these limits is hidden by injected ground truth.
