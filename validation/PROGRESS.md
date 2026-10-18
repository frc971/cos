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
