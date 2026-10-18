# Why density changes with the view

This diagnoses the frozen C++ implementation at `f1fd84e`; no production
algorithm or calibration changes are included. It reuses the same 200 frames,
five forward-translating camera poses, and original output hashes in
[audit.json](audit.json). Wider-angle invariance remains untested.

## Count and footprint both contribute

Select pile pairs whose independently rendered visible ball identities are
identical across all five views. This gives 96 ideal and 111 noisy pairs.
Their true counts, covariance footprints, and density ratios are constant.
For each pair, measure the maximum predicted ratio divided by its minimum.
One is perfectly stable; these are range spans over five views, not per-view
percentage errors.

| Diagnostic | Ideal median / p90 | Noisy median / p90 |
|---|---:|---:|
| Actual density ratio | 1.361 / 1.717 | 1.353 / 1.790 |
| Count ratio alone, holding footprint ratio fixed | 1.317 / 1.605 | 1.291 / 1.674 |
| Footprint ratio alone, holding count ratio fixed | 1.195 / 1.364 | 1.189 / 1.390 |
| Original tangent-sample footprint ratio | 1.111 / 1.293 | 1.086 / 1.279 |
| New footprint, detected positions snapped to reference positions | 1.173 / 1.351 | 1.180 / 1.330 |
| New footprint, all visible reference centers in place of detections | 1.021 / 1.045 | 1.016 / 1.037 |

Count and area spans are not additive: their changes can reinforce or cancel.
For the position-snapping diagnostic, use independent one-to-one matches within
15 cm across each frame; unmatched detections retain their original position.
The complete-reference diagnostic retains the original three-observation
covariance prior. Neither diagnostic supplies truth to the production tracker.
Recomputing the production area from its published points reproduces it within
3e-7 square meters.

The old tangent-projection centroid still has median error 4.7 cm ideal and
4.6 cm noisy across all matched observations. A useful centroid does not imply
unbiased per-ball sampling or a stable covariance. Replacing detected positions
with exact ones makes only a small improvement; changing which spatial centers
are represented makes a much larger difference. The density covariance blend
has introduced more footprint drift than the old tangent-sample covariance.

## Depth weighting exists, but is incomplete

Residual area uses approximately `Z^2 / (pi*r^2*fx*fy*norm(ray))`, and residual
arc length uses `Z*norm(ray) / (2*pi*r*sqrt(fx*fy))`. Thus distant pixels are
already weighted more heavily. The problem is not a missing distance factor.

There are two geometric shortcomings:

1. Residual pixels are intersected with the **ball-center plane**, treating
   sphere-surface pixels as center rays. Tangent projection includes the radius
   and surface normal; the residual projection does not. With camera height
   0.6267 m and ball-center height 0.075 m, the height difference is 0.5517 m.
   A surface-height offset `dz` changes the plane-assigned depth approximately
   by the factor `0.5517/(0.5517-dz)` relative to that surface point's optical
   depth. At `dz=+0.075 m`, this is 1.157, or 1.339 after squaring in the area
   weight. Those are possible local factors, not measured whole-pile errors.
   Occlusion changes which surface fragments survive, so the effect need not
   average away. It also stretches the projected evidence and can change its
   cluster ownership or rejection by the covariance gate.
2. Arc weights do not include the full directional projection Jacobian. An
   off-axis sphere has an elliptical projected outline. Integrating the current
   arc formula over an exact full outline, **even supplying perfect center
   depth**, produces the following apparent sphere-equivalents:

| Off-axis angle | Integrated arc evidence; expected approximately 1 |
|---|---:|
| 0 degrees | 1.0004 |
| 20 degrees | 1.0990 |
| 35 degrees | 1.3595 |
| 45 degrees | 1.7205 |

This numerical geometry check uses a 7.5 cm sphere at optical depth 4 m,
equal 1000 px focal lengths, 20,000 exact tangent-outline segments, and no
rasterization, distortion, occlusion, or edge-detector errors. It establishes
a normalization bias, not its isolated share of the observed pile drift.
The original contour-support weights use range-based inverse angular perimeter
and likewise do not fully correct directional projected arc length.

## The hybrid count has a discontinuous handoff

The predictor is `verified + 4.632368*unexplained_area + 2.421001*unexplained_arc`.
Accepting a sphere adds one verified count but removes its filled silhouette
and outline from the residual evidence. Those quantities were multiplied by
different calibrated coefficients; there is no conservation rule ensuring the
lost residual prediction equals the added one ball. Evidence from overlapping
spheres can also be covered by the accepted silhouette.

Consequently, a small detection-score change can change the total count sharply
even when the ball remains visible. For example, ideal `six_dense_steps-1009`,
reference pile 4, frames 0 and 1, has 22 visible balls in both frames. Verified
count rises from 9 to 10, but total predicted count falls from 14.74 to 12.76;
the residual contribution falls from 5.74 to 2.76. This is an observed example
of the unstable mixture, not an isolated experiment proving one acceptance
threshold caused the whole change.

Fixed pixel blur, Canny thresholds, outline-distance tolerances, vote smoothing,
and minimum unique-edge support also make candidate acceptance resolution and
contrast dependent. The upscaling step helps small balls but does not establish
scale invariance. Partial visible area or arc is not a fixed fraction of one
visible ball, so one global pair of regression weights cannot remove all of
this view dependence. The coefficients were fit to per-frame count error and
contain no cross-view consistency objective.

## Implications

The evidence supports retaining tangent projection for location, correcting
the arc normalization and residual-depth model, making the verified/residual
handoff consistent, and estimating the footprint without treating the changing
detected subset as an unbiased sample of the pile. Smoothing the ellipse alone
would leave substantial count drift. Exact attribution among count mechanisms
requires additional controlled runtime ablations; this diagnostic does not
claim a percentage breakdown between them.

Reference visibility is intentionally view dependent when balls gain or lose
all image evidence. A future angular sweep must report that separately from
estimator drift on fixed visible populations, as this analysis does.

Reproduce from downloaded metadata containing `ideal/` and `noisy/`:

```sh
python3 scripts/diagnose_gamepiece_density_views.py /tmp/ball-density-accepted \
  --output /tmp/view-diagnosis.json
```

The output includes per-input metadata hashes and the ablations:
[view-diagnosis.json](view-diagnosis.json).
