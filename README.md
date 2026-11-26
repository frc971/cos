cmake -B build -G Ninja -Wno-dev
cmake --build build

Initialize dependencies before configuring a fresh checkout:

```sh
git submodule update --init --recursive
```

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

`pathing::DijkstraSplineNode` accepts pose and goal channels, an output channel,
a navgrid path, and `SplineSettings`. It uses the same planner as
`DijkstraPathNode` and publishes a `SplinePathMessage` with the original route,
interpolation waypoints, cubic Hermite segments, visualization samples, and a
`SplineStatus` (`Ready`, `InvalidInput`, `NoPath`, or `FitFailed`). Segment
parameters run from 0 to 1. Spline tangents describe travel direction
independently of robot orientation.

Fitting starts with visibility-simplified waypoints. When a curve crosses an
obstacle or field boundary, the fitter inserts points along the original route
and refits. Clearance is certified by recursively subdividing Bézier control
hulls, rather than relying on visualization samples. Touching blocked cells or
an unresolved collision check is treated as unsafe. The navgrid must already
include any clearance needed for the robot's footprint.

Defaults are 32 refinement passes, at most 4096 fitting points, and samples at
most 0.02 meters apart along the curve. A same-position request produces a single
point and no segments. Exhausting fitting or sampling limits produces `FitFailed`
with the raw route retained and no usable spline or samples. Sampling is bounded
to 100000 points and subdivision depth 24; collision subdivision depth is 16.
Call `EnableSplineDebug(channel)` before registration to publish a
`streamer::PolylineMessage`, and register it using `NavgridStreamerNode::AddPolyline`.

`vision_bot_main` also supports live pathing from roboRIO NetworkTables pose and
goal inputs. See [the pathing contract](src/pathing/README.md) for topic names,
command fields, freshness requirements, and configuration flags. The roboRIO
publisher and drive-command consumer must be implemented separately.

Vision-bot's live pathing visualization is enabled by default at
`http://vision-bot:4975/navgrid`. Disable it with `--nonavgrid_streamer`, or change
its port with `--navgrid_streamer_port`.
