cmake -B build -G Ninja -Wno-dev
cmake --build build

Initialize dependencies before configuring a fresh checkout:

```sh
git submodule update --init --recursive
```

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

`pathing::DijkstraSplineNode` consumes a `PathingInputMessage` with the robot
pose and goal, plus a `streamer::NavgridMessage`. It fits a spline and publishes
the requested field-relative velocity as a `CommandMessage` each cycle.
Construct the navgrid from in-memory cells or load PathPlanner JSON with
`NavgridMessage::Load`. Reuse the same message while the grid is unchanged and
publish a new immutable snapshot when it changes.

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
Call `EnableSplineDebug(spline_channel, polyline_channel)` before registration
to publish a `SplinePathMessage` with the raw route, fitted cubic Hermite
segments, waypoints, and samples, plus a `streamer::PolylineMessage` for drawing.
Segment parameters run from 0 to 1, and spline tangents describe travel direction
independently of robot orientation. Register the spline with
`NavgridStreamerNode::AddPathingInputs` and the polyline with `AddPolyline`.
The streamer accepts a base `NavgridMessage` or a navgrid JSON path and field
image path. Use `AddNavgrid` for additional grid overlays from the control loop.

`vision_bot_main` also supports live pathing from roboRIO NetworkTables pose and
goal inputs. See [the pathing contract](src/pathing/README.md) for topic names,
command fields, freshness requirements, and configuration flags. The roboRIO
publisher and drive-command consumer must be implemented separately.

Vision-bot's live pathing visualization is enabled by default at
`http://vision-bot:4975/navgrid`. Disable it with `--nonavgrid_streamer`, or change
its port with `--navgrid_streamer_port`.
