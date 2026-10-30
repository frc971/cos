cmake -B build -G Ninja -Wno-dev
cmake --build build

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

## Second-bot ball pathing

`vision_bot_main` handles AprilTag localization and camera streaming only.

`second_bot_main` runs the grayscale ball U-Net on its gamepiece camera and
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

This uploads `models/ball_unet.engine` to
`root@10.9.71.11:/root/models/ball_unet.engine`, the service's configured model
path. To use a different engine, configure CMake with
`-DSECOND_BOT_ENGINE=/absolute/path/to/ball_unet.engine`.

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
