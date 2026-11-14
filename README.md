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
