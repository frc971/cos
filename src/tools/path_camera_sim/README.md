# path_camera_sim

Renders the 2026 field's 3D meshes from robot poses using the calibrated camera.
The renderer uses headless OpenGL ES 3 on dev-orin; it requires a working EGL
display and GPU. The field model is the official AdvantageScope 2026 asset in
`constants/field-cad` (see its `LICENSE`). Staged Fuel is omitted, matching
BOS when no gamepieces are configured. The tool writes timestamped PNGs and
`manifest.json`. `--fps` sets the frame rate (default 30); filenames are elapsed
seconds starting at zero, with six decimal places: `0.000000.png`,
`0.033333.png`, `0.066667.png`, etc. The manifest records `fps` and each frame's
unrounded `timestamp` in seconds. FPS must be finite, positive, and at most
1000000 to keep filenames unique at microsecond precision.

```sh
cd /root
./tools/path_camera_sim \
  --poses=/path/to/poses.json \
  --fps=30 \
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

To replace staged Fuel with Gaussian clusters, put a JSON array in `clusters.json`:

```json
[
  {"center_m":[5,3],"density":20,"covariance_m2":[0.5,0.08],"rotation_rad":0.7},
  {"center_m":[8,5],"density":10,"covariance_m2":[0.2,0.2],"rotation_rad":0}
]
```

Pass the file path with `--fuel_clusters`:

```sh
./tools/path_camera_sim --poses=/path/to/poses.json \
  --fuel_clusters=clusters.json \
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

The simulator renders one frame per pose; FPS sets timestamps and does not
resample the path or pace rendering. To render 50 frames spanning the whole
Quato path, sample its pose array uniformly, keeping both endpoints:

```sh
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/root/debug-output/quato-input')
poses = json.loads((root / 'Quato.poses.json').read_text())['poses']
sample = [poses[round(i * (len(poses) - 1) / 49)] for i in range(50)]
(root / 'Quato-50.poses.json').write_text(json.dumps({'poses': sample}))
PY
./tools/path_camera_sim \
  --poses=/root/debug-output/quato-input/Quato-50.poses.json \
  --fps=30 \
  --camera_config=constants/second_bot/left_camera.json \
  --fuel_clusters=/root/debug-output/quato-input/clusters.json \
  --fuel_seed=7 --fuel_noise=0.1 \
  --output_dir=/root/debug-output/quato-50-30fps
```

Run from `/root` on dev-orin. The original Quato pose file was found at
`root@build-server:/home/yasen/cos-worktrees/materials/Quato.poses.json`.
Its 351 poses cover seven path segments. The original, the sampled 50-pose file,
and an example `clusters.json` have been copied to the input directory above.

Each manifest embeds the camera calibration and lens-distortion setting.
Feed a run directly to `tools/hsv_cluster_visualizer --sim_run=/root/debug-output/path-camera-sim/entropic`
to generate cluster views and projected-point heatmaps using its exact poses.
Simulator output defaults to `/root/debug-output/path-camera-sim`. Visualizer
output defaults to `/root/debug-output/<run-name>/<layout>/clusters`.
Pass `--output_dir=DIR` to override either output directory.

For ground-truth validation, `--fuel_truth` adds `cluster_id` to generated Fuel
entries in cluster mode and `visible_fuel_pixels` to each frame. A separate
unlit integer-ID pass preserves opaque field/Fuel depth occlusion and applies
the same lens remap using nearest-neighbor sampling. Transparent field meshes
are omitted from the ID pass. These labels are not written into RGB images and
are never input to the tracker. Counts report visibility, including partial
silhouettes and image cropping; they are not assumed to reveal hidden balls.

Cluster manifests also record `cluster_centroids_m` (means of the actual generated
centers) and per-frame `visible_cluster_centroids`. A separate one-pixel point
pass tests these centers against opaque field depth and the camera frustum;
the distorted projection must also lie inside the output image. Fuel is omitted
from this pass so a ball does not hide its own pile's semantic center. As with
the ID pass, transparent field meshes do not occlude probes. These probes never
enter RGB or tracker input. They distinguish a visible centroid from a few
visible fragments whose centroid is behind a field structure or outside view.
