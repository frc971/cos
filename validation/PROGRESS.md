The source branch is `gamepiece-cluster-stability`; the original `ellipse-gamepiece`
worktree is untouched. Validation uses only path_camera_sim Gaussian Fuel layouts.

First checkpoint: built with `cmake --build build --parallel 6`, deployed with
`cmake --build build --target dev-orin`, seven focused tracker tests passed on
root@dev-orin. On the 17-frame ideal-clear approach, the baseline returned four
clusters initially, declining to two; the density tracker returned two distinct,
consistent IDs in every frame. JSON observations are in evidence/. These are
intermediate results: projection bias and partial-view shape stability still need
work, followed by varied ideal layouts and noisy simulations. No completion claim.

Run artifacts: root@dev-orin:/root/cluster-validation/{ideal-clear,baseline-clear,density-clear}.

Second checkpoint: sphere silhouette projection replaces all-column estimates.
Ten ideal simulator runs (compact, elongated, two groups, three groups, empty;
seeds 7 and 23) pass all 130 frame count comparisons, with no missed/extra groups
and no ID switches. Worst run p95 centroid error is 0.171 m; p95 covariance
Bures distance is 0.270 m. The evaluator uses generated ball positions and GPU
visibility IDs recorded separately from RGB. The detector never reads truth.
Next: positional-noise runs, entering/re-entering groups, neutral-zone culling,
and stronger end-to-end sphere/GPU tests. Not complete.
