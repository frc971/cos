# LiDAR circular scan view

`streamer::RPLidarScanStreamerNode` consumes `pathing::RPLidarMessage`
and serves a live 960×540 JPEG point plot through the existing MJPEG server.
Register it with `ControlLoop::RegisterNode` and pass the acquisition node's
output channel to its constructor. It does not publish control-loop messages.

Build and run the example on the robot:

```sh
cmake --build build --target rplidar_node
./build/examples/rplidar_node --config=constants/vision-bot/rplidar.json
```

When using the existing deployment layout, the executable is
`/root/examples/rplidar_node` and the config is
`/root/constants/vision-bot/rplidar.json`.

Open `http://<robot-host>:8080/lidar` directly in a browser. The scan view updates
for each scan, approximately 10 Hz in the example; console scan output remains
enabled. The browser must be able to reach the robot's HTTP port.

The sensor sits at the center of a circle, with each return drawn as a point
using its measured angle and distance. Zero degrees points up, and angles
increase clockwise (90° right, 180° down, 270° left), following the SDK angle
convention without a robot heading transform. Four labeled range rings provide
distance references. The outer circle defaults to 20 meters; adjust it with
`--scan_max_distance_m=20`. Set the HTTP port with `--scan_port=8080`.
The constructor also accepts a custom stream path.

Only the first `length` samples are considered, capped at the sample array size.
Zero-distance returns are drawn as yellow points on the outer circle at their
published angles and counted separately. These markers indicate missing distance
returns, not measured obstacles at maximum range. Returns beyond the
configured range are also counted separately; the exact maximum is drawn on the
outer circle. There is no quality threshold. Points describe the current scan,
without temporal accumulation, using equal distance scales in both directions.
Missing scans display “No scan available”; scans with no usable returns display
“No usable returns” while still showing any zero-distance markers. The image
reports sample, plotted, zero, and over-range counts; “Plotted” counts valid
distance points and “Zero” counts the yellow markers separately. JPEG failures
are logged and the next scan retries publication.
