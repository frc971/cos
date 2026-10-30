cmake -B build -G Ninja -Wno-dev
cmake --build build

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

## Second-bot ball pathing

`vision_bot_main` handles AprilTag localization and camera streaming only.

`second_bot_main` runs the RGB ball U-Net on its gamepiece camera and
publishes a robot-relative translation goal. The internal control-loop channel
is `ball_pathing/velocity_goal`. NetworkTables publishes
`/COS/BallVelocityGoal` as:

```text
[vx_mps, vy_mps, target_x_m, target_y_m, estimated_ball_count,
 valid, held, camera_timestamp_seconds, projected_ball_evidence_m2]
```

Robot coordinates are +X forward and +Y left. When no cluster is visible, the
last goal is marked `held` for 0.5 seconds; it then becomes invalid with zero
velocity. Heading remains under roboRIO control.

Velocity favors lateral correction: `vx = 0.75 * target_x` and
`vy = 1.5 * target_y`, with robot-relative distances in meters and velocity in
meters per second. For nonzero target offsets, the weighted vector is scaled
to a total speed between 1.0 and 1.5 m/s so the robot keeps moving into the
cluster instead of stopping near its center. Tune the gains using
`--ball_forward_velocity_gain` and `--ball_lateral_velocity_gain` (inverse
seconds). These replace the previous `--ball_velocity_gain` flag. Configure
speed limits with `--ball_minimum_speed_mps` and `--ball_maximum_speed_mps`;
`--ball_stop_radius_m` has been removed. If the target offset is exactly zero,
there is no direction to command, so the velocity is zero. The existing
lost-target hold and invalid-goal stop behavior remain unchanged.

The segmentation mask is sampled and projected onto the floor in robot
coordinates. Each sample is weighted by its projected floor area, which
compensates for perspective so nearby large masks do not automatically beat
distant connected piles. A circular window over this density grid selects the
largest concentration of ball evidence; connected balls therefore contribute
their full segmented area rather than being collapsed into one component.

The prebuilt TensorRT engine and ONNX model are committed under `models/`;
see [model compatibility and validation](models/README.md). The engine is
hardware-specific. Deploy the committed engine together with the binaries,
constants, and service files:

```sh
cmake --build build --target second-bot
```

This uploads `models/ball_unet_rgb.engine` to
`root@10.9.71.11:/root/models/ball_unet.engine`, the service's configured model
path. To use a different engine, configure CMake with
`-DSECOND_BOT_ENGINE=/absolute/path/to/ball_unet.engine`.

For existing build directories, select the new RGB default explicitly:

```sh
cmake -S . -B build -DSECOND_BOT_ENGINE="$PWD/models/ball_unet_rgb.engine"
```

`--ball_pathing_input_mode=rgb` is the runtime default. The decoder produces
interleaved RGB, and GPU preprocessing resizes it into three normalized float
planes. The node's constructor receives an explicit input mode and checks that
the engine matches it. To use the grayscale model, deploy
`models/ball_unet.engine` via `SECOND_BOT_ENGINE` and change the service flag to
`--ball_pathing_input_mode=grayscale`. `ball_pathing_disk_replay` accepts the
same input-mode flag, alongside its `--model_path` option.

To regenerate an engine:

```sh
python3 -m venv .cache/unet-export
.cache/unet-export/bin/pip install -r scripts/unet-export-requirements.txt
PYTHON_BIN=.cache/unet-export/bin/python scripts/export_unet_engine.sh
```

The exporter retrieves the pinned checkpoint from `yasencomp`, verifies its
SHA-256, builds FP16 on vision-bot, installs
`/root/gamepiece_models/ball_unet.engine`, and downloads a local copy. Pass a
checkpoint and output path explicitly to override the defaults. Set
`--ball_pathing_model=` to disable the pipeline for diagnostics.
Set `INPUT_MODE=rgb` when exporting the three-channel model; its default local
output is `ball_unet_rgb.engine`.
