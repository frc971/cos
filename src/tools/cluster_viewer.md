`cluster_viewer` runs the gamepiece tracker and previews the exact densest-cluster
publisher output, without publishing to the robot. Main programs are unchanged.

Live camera, with the robot's `/Drive/Pose` from NetworkTables:

```sh
/root/tools/cluster_viewer --config=/root/constants/second_bot/color.json --nt_server=10.9.71.2
```

Replay a timestamped JPEG folder with a fixed field pose (meters, radians):

```sh
/root/tools/cluster_viewer --config=/root/constants/gamepiece/camera.json --log_path=/path/to/frames --pose=2,4,0
```

Open these URLs (default `--port=5805`):

- `http://dev-orin:5805/stream`: current 2-sigma cluster ellipses and densities on
  the field, following the cluster pipeline test's field overlay.
- `http://dev-orin:5805/field`: the selected ellipse, center, and four axis
  endpoints from the actual publisher's ten-number output.
- `http://dev-orin:5805/hsv`: the exact binary HSV threshold mask used by the tracker.

The field uses the AprilTag layout's origin, a one-meter grid, +X right and +Y up.
Without a pose the HSV stream remains available and field views show a waiting
message. A fixed pose also works with the live camera. Replay ends at EOF; use
Ctrl-C to stop a live run. Frames are JPEG encoded only while a stream has clients.
