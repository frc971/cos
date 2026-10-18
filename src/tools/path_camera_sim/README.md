# path_camera_sim

Renders the 2026 field's 3D meshes from robot poses using the calibrated camera.
The renderer uses headless OpenGL ES 3 on dev-orin; it requires a working EGL
display and GPU. The field model is the official AdvantageScope 2026 asset in
`constants/field-cad` (see its `LICENSE`). Staged Fuel is omitted, matching
BOS when no gamepieces are configured. The tool writes `frame_000000.png`,
etc., and `manifest.json`.

```sh
cd /root
./tools/path_camera_sim \
  --poses=/path/to/poses.json \
  --camera_config=constants/second_bot/left_camera.json \
  --output_dir=/root/debug-output/path-camera-sim
```

The pose file is a JSON array or an object containing a `poses` array:

```json
{
  "poses": [
    {"translation_m": [1.0, 2.0], "rotation_rad": 0.0},
    {"translation_m": [1.1, 2.0], "rotation_rad": 0.1}
  ]
}
```

Coordinates are WPILib field meters and radians. Three-dimensional localization
poses can use a three-element `translation_m` and a `rotation_rpy_rad` array.
Use `--field_dir` to select another AdvantageScope field directory containing
`config.json` and `model.glb`.

To render Fuel at its staged positions followed by a reproducible scattered
layout, add `--fuel_entropy=0.4 --fuel_seed=7`. The tool writes the two runs to
`<output_dir>/rigid` and `<output_dir>/entropic`, each with its own frames and
manifest. The manifest records every Fuel position. Entropy ranges from 0 to 1:
at 0 all 456 pieces keep their staged positions; at 1 each survivor has up to
2 m standard deviation of horizontal displacement and each piece has a 50%
removal chance. The placement and random sampling follow BOS's Fuel generator.
The staged Fuel meshes are already embedded in `model.glb`, so no separate
gamepiece asset is required.

To replace staged Fuel with Gaussian clusters, pass a JSON array:

```sh
./tools/path_camera_sim --poses=/path/to/poses.json \
  --fuel_clusters='[{"center_m":[5,3],"density":20,"covariance_m2":[0.5,0.08],"rotation_rad":0.7},{"center_m":[8,5],"density":10,"covariance_m2":[0.2,0.2],"rotation_rad":0}]' \
  --fuel_noise=0.1 --fuel_seed=7
```

Each center uses WPILib field meters. Density is the peak balls per square meter
of the Gaussian before noise; the two positive covariance values are variances
along its principal axes in square meters (standard deviations are their square
roots). Rotation is counterclockwise from field +X, in radians. Each cluster
samples `round(2*pi*density*sqrt(v1*v2))` balls. `--fuel_noise` adds independent
Gaussian displacement to X and Y, with standard deviation in meters (default 0).
Balls outside the field boundary are discarded; overlaps and obstacles are not
resolved. At most 100000 balls are allowed. Density 0 or an empty array produces
no balls. Use `--fuel_seed` for repeatable layouts.

Cluster mode writes one run directly to `--output_dir`, with cluster parameters,
noise, seed, and generated positions in its manifest. It cannot be combined with
`--fuel_entropy`, which selects the staged/scattered comparison mode.

Each manifest embeds the camera calibration and lens-distortion setting.
Feed a run directly to `tools/hsv_cluster_visualizer --sim_run=/root/debug-output/path-camera-sim/entropic`
to generate cluster views and projected-point heatmaps using its exact poses.
Simulator output defaults to `/root/debug-output/path-camera-sim`. Visualizer
output defaults to `/root/debug-output/<run-name>/<layout>/clusters`.
Pass `--output_dir=DIR` to override either output directory.
