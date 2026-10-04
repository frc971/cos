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
  --output_dir=sim-output
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
