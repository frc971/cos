cmake -B build -G Ninja -Wno-dev
cmake --build build

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

`navgrid_streamer_test` runs a simulated Dijkstra follower and streams obstacles
in red and its grid route in green at `http://localhost:4971/navgrid`. Tune route
selection with `--turn_penalty=0.3` (the default), or set it to `0` for distance
only. The cost is distance in meters plus this weight times the total change in
travel direction in radians; for example, a 90-degree turn adds about 0.47 meters
at the default weight. Larger weights can select longer, straighter routes.
The first move has no turn penalty, and robot heading is controlled independently.
The follower still skips to the furthest visible waypoint, so the penalty
optimizes the grid route rather than the exact motion after waypoint skipping.
Use `--start_x`, `--start_y`, `--goal_x`, and `--goal_y` to choose positions in
meters, and `--start_heading` / `--goal_heading` for headings in degrees.

The demo also runs `pathing::DijkstraSplineNode` and draws its fitted curve in
blue, with fitted control waypoints highlighted as magenta circles. This node
accepts pose/goal channels, an output channel, a navgrid path,
and `SplineSettings`. It performs its own search using the same planner as
`DijkstraPathNode`, then publishes `SplinePathMessage`: the original route,
interpolation waypoints, evaluable cubic Hermite segments, visualization samples,
and a `SplineStatus` (`Ready`, `InvalidInput`, `NoPath`, or `FitFailed`). Segment
parameters run from 0 to 1; spline tangents describe travel direction and are
independent of robot orientation. The demo still moves using the original
velocity controller; the spline node publishes geometry without timing or
velocity commands.

A cyan arrow at the simulated pose shows the current target translation velocity
in field coordinates. Its length represents one second of commanded motion
(1 m/s draws a 1-meter arrow); it shrinks to a dot when translation stops.

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
