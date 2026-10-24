# Cluster NetworkTables interoperability

`streamer::ClusterRioStreamerNode` consumes a tracker publication and a
`localization::PositionEstimateMessage` from the same context. Register it with
the loop that contains those publications and link
`cos::cluster_rio_streamer_node`:

```cpp
auto sender = std::make_shared<streamer::ClusterRioStreamerNode>(
    "clusters", "pose", "/COS/Clusters");
loop.RegisterNode(sender);
```

The node waits for both dependencies (including explicit null messages), then
publishes once per context. Its context output at `/COS/Clusters` contains
`ValueMessage<std::vector<gamepiece::cluster_nt_t>>`. A null frame produces a
null context output without transmitting a new observation. A real empty frame
or a null pose transmits an empty array to clear old targets. Clusters without
ground coordinates or with nonfinite/invalid geometry or density are omitted.
The center and covariance must come from the tracker's robot-relative output;
the sender transforms them using the pose in this context.

The wire type is `struct:ClusterNT[]`, with 48 bytes per cluster. These six
little-endian doubles appear in this exact order:

| Field | Meaning |
| --- | --- |
| `center_x`, `center_y` | Field coordinates, meters |
| `field_relative_semimajor_axis_yaw` | Field yaw, radians in [0, pi) |
| `semimajor_length`, `semiminor_length` | 2-sigma radii from ground covariance, meters |
| `predicted_density` | Tracker `density`, predicted visible balls/m² |

There is no array length prefix or native C++ struct copy. WPILib's struct-array
API handles packing and automatically advertises `/.schema/struct:ClusterNT`.
`ClusterNT.java` is the matching immutable Java record and `Struct` implementation;
copy it into the robot package when adding the robot-side consumer. Retain the
subscriber for its entire useful lifetime and read its queue each robot tick.
WPILib 2026.2.1's typed
[`StructArraySubscriber.readQueue()`](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/ntcore/src/main/java/edu/wpi/first/networktables/StructArrayEntryImpl.java)
drops empty arrays and mishandles batches containing them. Use its `StructBuffer` with the exact
struct-array type to preserve empty observations and their timestamps:

```java
var buffer = StructBuffer.create(ClusterNT.struct);
var clusters = instance.getRawTopic("/COS/Clusters")
    .subscribe("struct:ClusterNT[]", new byte[0], PubSubOption.periodic(0.001),
        PubSubOption.sendAll(true), PubSubOption.keepDuplicates(true),
        PubSubOption.pollStorage(64));
// In periodic():
for (var sample : clusters.readQueue()) {
  ClusterNT[] values = buffer.readArray(sample.value);
  // sample.timestamp is NT time in microseconds, including empty arrays.
}
```

Both ends request 1 ms, which WPILib NT4 clamps to a 5 ms minimum (200 Hz).
Requesting exactly 5 ms instead rounds up to 10 ms in WPILib 2026's client.
`sendAll` prevents coalescing observations; `keepDuplicates` preserves unchanged
observations. This increases the transmission rate; it does not invent frames
when the camera/tracker is slower. NetworkTables' default period is 100 ms.
The test deliberately does not call `Flush()` per frame.

## Running the real C++ to Java test

Build, check clangd, and deploy per the repository instructions. On an x86-64
Linux host with JDK 17 or newer (the example robot uses WPILib 2026.2.1), run:

```sh
JAVA_HOME=/path/to/jdk src/tests/java/run_cluster_interop.sh
```

This compiles the two Java files with `javac -Xlint:all -Werror` independently
of both CMake and `~/971-second-robot-2026`. The script caches official WPILib
Java jars and Linux JNI libraries under `/tmp/cos-cluster-java`; override with
`COS_JAVA_TEST_CACHE`. It listens on port 15810 (`COS_NT_TEST_PORT` overrides it).
After `JAVA_READY`, run the actual repository sender on the Orin, using a host
address reachable from the device:

```sh
ssh root@dev-orin '/root/tests/cluster_nt_java_publisher HOST_ADDRESS 15810'
```

The Java server uses WPILib's `StructBuffer` to decode all queued wire values;
it also checks that a typed `StructArraySubscriber` decodes the final nonempty
array. The C++ unit test uses a raw queue for empty-array assertions because
WPILib 2026's typed C++ queue reader also discards zero-length arrays.

The Java server asserts all 1,000 ordered samples, all six fields, empty and
large arrays, duplicate preservation, increasing NT timestamps, the exact topic
type, and the schema received from C++. It checks both sample delivery above
150 Hz and distinct queue-arrival batches above 110 Hz, so receiving 200 samples
per second in 100 Hz batches cannot pass. It acknowledges success to C++ only
after verification. Run it several times to check repeatability.

Run `/root/tests/cluster_rio_streamer_node_test` on the device for geometry,
dependency arrival order, simultaneous callbacks, invalid inputs, and stale
target clearing. This is supplementary to the separate-process Java test.

## Verified results

Validated on 2026-10-23 using the AArch64 repository build on `dev-orin` as the
NT4 client, and the separately compiled Java stub on the Linux development host
as the server, over Tailscale. Java used JDK 21 and WPILib 2026.2.1.

| Run | Arrays verified | Sample rate | Distinct arrival batches |
| --- | --- | --- | --- |
| 1 | 1,000 | 200.1 Hz | 194.3 Hz |
| 2 | 1,000 | 199.9 Hz | 194.5 Hz |
| 3 | 1,000 | 199.9 Hz | 194.3 Hz |

Each run included 26 empty arrays and 26 arrays of 128 clusters; all schemas,
fields, ordering, duplicate observations, and timestamps passed. On the same
device, the three sender unit tests passed 20 repetitions and all 20 tracker
tests passed, including padded GPU-frame processing. `./scripts/build.sh` and
clangd warning/clang-tidy checks passed. For the GoogleTest translation unit,
`--tweaks=none` disabled clangd's failing ExtractFunction refactoring self-tests;
compiler diagnostics and clang-tidy remained enabled.

## References

The implementation follows the explicit serializer and immutable Java struct
conventions in WPILib's
[C++ Translation2d serializer](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/wpimath/src/main/native/include/frc/geometry/struct/Translation2dStruct.h)
and [Java Translation2d serializer](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/wpimath/src/main/java/edu/wpi/first/math/geometry/struct/Translation2dStruct.java).
For FRC usage, the local 971 project's
`src/main/java/frc/robot/subsystems/drive/Telemetry.java` retains typed struct-array
publishers, and [6328's Northstar consumer](https://github.com/Mechanical-Advantage/RobotCode2025Public/blob/main/src/main/java/org/littletonrobotics/frc2025/subsystems/vision/VisionIONorthstar.java)
retains subscribers and consumes timestamped queues with `sendAll` and
`keepDuplicates`. See also WPILib's
[publish/subscribe options](https://docs.wpilib.org/en/stable/docs/software/networktables/publish-and-subscribe.html)
and the [NT4 client period calculation](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/ntcore/src/main/native/cpp/net/ClientImpl.cpp).
