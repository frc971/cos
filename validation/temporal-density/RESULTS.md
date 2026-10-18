# Persistent density estimation

Worktree: `/root/cos-worktrees/gamepiece-temporal-density`. Baseline: `f1fd84e`,
the frozen per-frame density implementation. This extends that method with
stationary field-space hypotheses; it does not replace the pile separation
algorithm. The canonical scalar remains `density = count / effective_area`.

## What persists between frames

Each supported sphere supplies a field-space position. Before searching for
new spheres, the next frame projects those positions through the current robot
pose and camera calibration, tests their predicted arcs against current image
edges, and refines their positions. Birth proposals use unexplained edges.
Existing hypotheses have a lower reacquisition threshold, but still require
independent current edges. The tracker consolidates positions over observations
instead of replacing them with every new contour fit.

Out-of-view positions remain available for re-entry. Missing sensor frames are
not negative evidence. Nine unsupported in-view observations retire a sphere;
1,000 processing calls without support expire old map entries even outside view.
Blank images publish no clusters. Switching between field and robot coordinates
resets the map. Without a pose, the model assumes a stationary camera.

The footprint combines consolidated centers, briefly missed centers, and the
original tangent-sample covariance. A per-pile count and covariance posterior
persists in **field coordinates**, so rotating the camera does not rotate the
stored physical pile. Ordinary updates use gain 0.20. After three consecutive
large, same-direction count innovations, gain 0.65 remains active until the
estimate approaches the new evidence. This retains continuity while allowing
real additions and removals. These are frame-based thresholds, not timestamps.

This is more than smoothing the outputs of independent detections: remembered
positions change current proposals, edge ownership, depth assignment, and the
spatial footprint. `--density-filter-only` disables those predictions while
retaining the posterior update; `--density-no-history` disables both.

## Geometry and count experiments

Arc evidence now uses the directional projected-sphere Jacobian, including
unequal focal lengths and off-axis rays. Integrating a full projected sphere
outline is checked by a C++ test at several angles. Residual pixels use local
candidate-center depth when available and pile-center depth otherwise, rather
than assigning every surface pixel its intersection with the center plane.
The approximate sphere-area correction includes depth squared and off-axis
projection. Weak, image-supported candidates supply depth without being
counted as verified spheres.

Two count models were tried. The retained model combines verified spheres with
calibrated residual area and edges. The alternative jointly fits verified
count and **total** image evidence, avoiding residual-removal discontinuities.
Although steadier, that alternative reduced pilot >=2x ordering accuracy to
60.5%, increased median count error to 21.8%, and failed the original count
tests. It remains available as `--density-total-count` for further comparison.
The rejected pilot precedes the final memory-expiry and settling refinements.
[Failed tests](experiments/total-tests.txt),
[pilot accuracy](experiments/total-pilot-accuracy.json), and
[pilot temporal results](experiments/total-pilot-temporal.json) preserve that
rejection. The residual predictor's handoff is still approximate; persistence
reduces its instability rather than making its single-frame model exact.

Calibration excludes all validation seeds. It uses the same 256 training
frames as the previous method; ambiguous pre-existing cluster splits are
excluded from fitting and retained in evaluation. The final coefficients and
feature hashes are recorded in [calibration.json](calibration.json).
The retained instantaneous predictor is:

```
count_observation = verified + 4.305261 * residual_area + 4.437539 * residual_arc
```

The two coefficients are nonnegative, with no intercept. The fit uses 1,458
unambiguous pile observations and has median absolute count error 15.8%.
Raw evidence and verified positions are independent of the count coefficients;
the training reports store those features for reproducing the fit.

## Validation

The forward test reuses the previous 100 ideal and 100 noisy frames. Pile pairs
with identical visible ball identities across all five poses are the strict
cross-view stability cohort. The angular test uses `angular_poses.json`:
25 positions on a 4.2 m arc, yaw -25.8 to +25.8 degrees, with 3.65 m lateral
travel. `angular_scenarios.json` includes four six-pile layouts and a seven-pile
layout, each with seeds 5003 and 6007, giving 250 frames. These seeds are excluded
from calibration. Physical viewing direction changes as well as camera yaw.

Across that wide path, visibility changes in every pile-pair cohort; no pair
keeps exactly the same visible population throughout all 25 frames. Angular
jitter is therefore measured on adjacent pairs of frames where both piles'
visible ball identities are unchanged. Longer-range drift normalized by the
per-view reference is reported separately, including its failures. Missing or
extra clusters remain in the accuracy reports and are not silently dropped.

Noisy scenes perturb ball positions by sigma 0.1 m; positions remain fixed
across frames, and camera poses and calibration are exact. The reference counts
every ball with at least one independent ID-render pixel, including partial
balls, and excludes zero-evidence balls. Simulator labels never enter tracking.

### Forward motion

Range variation means `max(predicted pile ratio)/min(predicted pile ratio)-1`
across the five views. Adjacent variation uses the multiplicative change between
successive views, expressed as a positive percentage. These measure consistency,
separately from accuracy against the reference.

| Mode | Ideal median range variation | Ideal median adjacent variation | Noisy median range variation | Noisy median adjacent variation |
|---|---:|---:|---:|---:|
| Previous method | 36.1% | 12.2% | 35.3% | 10.9% |
| Normalized, no history | 41.8% | 13.8% | — | — |
| Normalized, filter only | 14.4% | 3.4% | — | — |
| **Persistent hypotheses + posterior** | **10.1%** | **2.3%** | **11.5%** | **2.0%** |

The strict range cohorts contain 96 ideal and 111 noisy pile pairs. Adjacent
cohorts contain 755 and 750 comparisons. The new range variation at the
90th percentile is 33.8% ideal and 31.2% noisy, versus 71.7% and 79.0% before.
Geometry corrections alone did not improve consistency. Predicting hypotheses
adds a benefit beyond the identical posterior filter applied to independent
frame detections. After the first frame, 89.2% / 89.7% of accepted positions
reuse the map (ideal / noisy).

| Accuracy over all 600 matched pile observations per condition | Previous ideal | New ideal | Previous noisy | New noisy |
|---|---:|---:|---:|---:|
| Median absolute count error | 16.0% | 16.7% | 15.6% | 16.9% |
| Median density-ratio error factor | 1.43 | 1.45 | 1.42 | 1.39 |
| Correct ordering for >=2x reference contrast | 85.0% | 85.9% | 85.5% | 88.9% |

Count accuracy is slightly worse in these conditions, while relative ordering
is retained or improved. The smoother outputs are still approximate estimates.
Reports: [ideal](ideal-temporal.json), [noisy](noisy-temporal.json),
[no history](ideal-stateless-temporal.json), [filter only](ideal-filter-temporal.json).

### Wider angular path, including seven piles

All modes process identical RGB. There are 2,762 adjacent comparisons with
unchanged visible identities in both piles, and 829 within-frame comparisons
with reference contrast >=2x.

| Mode | Median adjacent variation | 90th percentile | >=2x ordering accuracy | Median count error |
|---|---:|---:|---:|---:|
| Previous method | 9.7% | 30.0% | 93.4% | 16.2% |
| Normalized, no history | 10.5% | 33.0% | 91.9% | 15.0% |
| Normalized, filter only | 2.6% | 7.4% | 92.4% | **12.8%** |
| **Persistent hypotheses + posterior** | **2.1%** | **6.1%** | **93.6%** | 15.5% |

91.0% of accepted positions after the first frame reuse field-space hypotheses.
Filter-only has better count accuracy here; predictive mode has better
continuity and ordering. The median full-path range of reference-adjusted ratio
error improves from factor 2.16 to 1.52, but its new 90th percentile is still
2.38. Therefore the result establishes improved continuity, not complete
viewpoint invariance. The median per-frame ratio error factor is 1.31, versus
1.29 before. Reports: [previous](angular-baseline-temporal.json),
[no history](angular-stateless-temporal.json), [filter only](angular-filter-temporal.json),
[persistent](angular-temporal.json).

There are 1,475 image-visible reference group observations, of which 1,435
match predictions. The original extractor still misses 40 groups and creates
four extra outputs, affecting 36 frames. These failures are identical to the
baseline. Forward runs likewise retain five ideal and nine noisy extra outputs.
[Geometry regression](geometry-regression.json) checks all pre-existing geometry,
IDs, and support fields for exact equality on all 1,150 processed frames across
the model comparisons (450 unique input frames). This density update does not
repair the extractor's missing groups or splits.

Median processing time on dev-orin is 132.8 ms ideal, 135.9 ms noisy, and
137.9 ms angular; corresponding 95th percentiles are 163.0, 165.1, and 168.5 ms.
Image loading and drawing are excluded. The old angular timing was collected
alongside training, so it is not used for a speedup claim.

### Controlled response and C++ tests

All **34 tests pass**: 20 tracker, nine image utilities, and five ellipse tests.
[Target log](target-tests.txt), [tracker XML](unit-results.xml).
The tests cover exact off-axis arc mass (within 0.3% over the tested angles and
unequal focal lengths), look-away/re-entry, blank output, in-view expiry,
coordinate resets, camera yaw, and real population changes.

In the 25→9→25-ball image sequence, the estimate is 27.8 before removal, 12.3
after six removal frames, and 26.9 after six addition frames. It responds rather
than freezing at the first count; the remaining count bias is visible. In the
stationary-pile yaw test, footprint area varies by 1.9% across the tested views.
These are simulator checks; real-camera and odometry-error validation remain
necessary for deploying the estimator on a moving robot.

## Visual review

Open [continuity.html](continuity.html) for the standalone interactive comparison:
fixed-population forward ratios, wider-angle ratios against the changing visible
reference, and the controlled count response. The forward pair was selected
using the previous method's median variation, before inspecting the new result.
Unavailable ratios appear as gaps. Plotly 6.3.1 generates the chart, with its
JavaScript embedded for offline viewing.

The same six-pile example used in the previous report remains available:
[camera](six-pile-camera.png), [field](six-pile-field.png). Each pile has
20 image-visible balls at that frame, with different spatial spreads.
The seven-pile review uses the fixed middle frame (12) of `seven-5003`:
[camera](seven-pile-camera.png), [field](seven-pile-field.png).
White field points show all generated centers; colored rings show supported
individual positions. Camera and field labels show the canonical density.
Drawn ellipses use the original tracker geometry; `density_area_m2` stores the
posterior effective area. Heatmaps continue to visualize contour support.

## Reproduce using C++

```sh
cd /root/cos-worktrees/gamepiece-temporal-density
cmake --build build --parallel 6
cmake --build build --target dev-orin
ssh root@dev-orin
cd /root
./tools/hsv_cluster_visualizer \
  --sim_run=/root/cluster-validation/temporal-density/accepted/angular/seven-5003 \
  --output_dir=/root/cluster-validation/temporal-density/review-replay
```

Inspect `frame_000012_clusters.png`, `frame_000012_field.png`, and `clusters.json`.
Each input advances the model once; reproducing frame 12 requires processing
the preceding frames. The validated binary is preserved under
`accepted/tools/hsv_cluster_visualizer`. Its SHA-256 is recorded in
[audit.json](audit.json), alongside source and input/report hashes.
The `density_history` and `density_count_model` JSON fields identify each mode;
instantaneous count/area and prediction/reuse counts are diagnostic fields.

The complete target replays are under `accepted/{ideal,noisy,angular}`.
Corresponding `*-stateless` and `*-filter` directories contain the ablations.
The exact run script is preserved under `accepted/review/run-temporal-density-accepted.sh`.
That script replays existing immutable RGB and writes to the accepted output
directories; change its output root to retain a second independent copy.

Fetch metadata locally and regenerate the angular reports:

```sh
rsync -r --include='*/' --include='manifest.json' --include='clusters.json' \
  --exclude='*' root@dev-orin:/root/cluster-validation/temporal-density/accepted/ \
  /tmp/temporal-density-accepted/
python3 scripts/evaluate_gamepiece_density.py /tmp/temporal-density-accepted/angular \
  --output /tmp/angular-accuracy.json
python3 scripts/evaluate_gamepiece_density_temporal.py /tmp/temporal-density-accepted/angular \
  --report /tmp/angular-accuracy.json --output /tmp/angular-temporal.json
python3 scripts/calibrate_gamepiece_density.py \
  validation/temporal-density/training-development.json \
  validation/temporal-density/training-extended.json --output /tmp/calibration.json
```

The evaluators require NumPy and SciPy. Regenerate the optional chart with
Plotly installed using `scripts/plot_gamepiece_density_temporal.py` and the same
metadata. Baseline, failed pilots, and final replays remain separate on the
target. The accepted source and outputs are identified by the audit and the
`SOURCE_COMMIT` file in the target review directory.
