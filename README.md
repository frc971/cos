# COS

COS runs on the robot’s NVIDIA Jetson Orin. It captures camera frames, uses
AprilTags to estimate the robot’s pose on the field, and sends that estimate to
the roboRIO.

## How to Build

Use CMake 3.23 or newer, Ninja, and a Jetson Orin build environment with
C++23 and CUDA C++23 support. CMake needs to find the CUDA Toolkit, JPEG,
OpenCV, WPILib, Eigen3, NVIDIA VPI, and libuvc. When building against a Jetson
sysroot, pass your toolchain file to the configure command with
`-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake`.

Run from the repository root.

```sh
git submodule update --init --recursive
cmake -B build -G Ninja -Wno-dev
cmake --build build --parallel 6
```

The robot executable is `build/main/second_bot_main`. Run the registered tests
with `ctest --test-dir build --output-on-failure`.

To deploy a build for hardware testing, run the following after building. The
target uses SSH and rsync to upload binaries, libraries, and configuration to
`root@dev-orin`.

```sh
cmake --build build --target dev-orin
```

## Code structure

[`src/control_loop/`](src/control_loop/) contains the scheduler and the node,
context, and message interfaces. Executables in [`src/main/`](src/main/)
assemble pipelines from modules such as [`src/camera/`](src/camera/),
[`src/apriltag/`](src/apriltag/), and [`src/localization/`](src/localization/).
Robot-specific configuration lives in [`constants/`](constants/).

### What is a node

A node implements [`control_loop::INode`](src/control_loop/node.h). It declares
the named channels and message types it consumes (`GetDependencies()`) and
produces (`GetPublications()`). `CreateCallback()` supplies the work to run
for a context. `RegisterCallback()` stores downstream callbacks that the node
invokes after making its results available.

This node reads an integer from `input`, doubles it, and publishes the result
on `doubled`. If the upstream publisher has no value for the current cycle, the
node publishes `nullptr` and still notifies its subscribers. Register it with
`RegisterNode()` in a loop that has a publisher for `input`.

```cpp
#include <memory>
#include <vector>

#include "control_loop/node.h"

class DoubleNode final : public control_loop::INode {
 public:
  using Context = control_loop::Context;
  using Callback = std::function<void(const Context&)>;
  using Number = control_loop::ValueMessage<int>;
  using Channels = std::vector<control_loop::MessageDescriptor>;

  auto CreateCallback() -> Callback override {
    return [this](const Context& context) {
      const auto* input = context->GetMessage<Number>("input");
      std::shared_ptr<Number> output;
      if (input != nullptr) {
        output = std::make_shared<Number>(input->value * 2);
      }
      context->SetMessage("doubled", output);
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };
  }

  auto GetDependencies() const -> const Channels& override {
    return dependencies_;
  }

  auto GetPublications() const -> const Channels& override {
    return publications_;
  }

  void RegisterCallback(const Callback& callback) override {
    callbacks_.push_back(callback);
  }

 private:
  Channels dependencies_{{"input", typeid(Number)}};
  Channels publications_{{"doubled", typeid(Number)}};
  std::vector<Callback> callbacks_;
};
```

### What is a control loop

[`ControlLoop`](src/control_loop/control_loop.cc) connects publishers to
subscribers and starts new cycles on its own thread. Register entry-point nodes
(e.g. the camera that will be serving frames which the rest of the loop is based on)
with `RegisterDependencyNode()` and downstream nodes with `RegisterNode()`
before calling `Start()`. A dependency node is called directly for each new
context. A downstream node is called by the publishers it subscribes to.

The period passed to the constructor is how long the loop sleeps after scanning
all context slots. `SetMaxContext()` controls how many cycles can be in flight
at once. A slot stays occupied while any work still holds that cycle’s context.

### What is a context

A [`Context`](src/control_loop/context.h) is a shared pointer to a single
cycle’s state. Every new context has a unique ID within the loop, a start time,
and a fresh, mutex-protected map from channel names to messages. Each channel
keeps the first entry written in that cycle. The same channel name can hold a
different value in every context.

`Exists(channel)` checks whether an entry has been written, including a
`nullptr` entry. `GetMessage<T>(channel)` returns a pointer to the typed value
or `nullptr` if there is no value of that type. Hold the context while using
that raw pointer. `GetSharedMessage<T>()` gives the caller shared ownership of
the message itself.

### Full pipeline

1. **Wire the graph.** `Start()` checks that each channel has at most one
   publisher, each publication declares a single type, and every downstream
   dependency has a publisher of an accepted type. For each dependency it
   registers the subscriber’s callback with that publisher.
2. **Start a cycle.** The loop thread scans its slots. It creates a new context
   for each empty slot or slot whose old context is owned only by the loop (meaning that all   downstream nodes have relinquished control and it can be discarded).
   It invokes the dependency nodes in registration order, then invokes any
   callbacks registered directly on the loop. It stores the new context in
   the slot and sleeps for the configured period after scanning all slots.
   A scan can start multiple cycles when several slots are available.
3. **Propagate results.** A publisher writes to the current context and calls
   its subscribers with that same context. Those callbacks can run inline or
   be invoked from worker threads. A node that needs inputs from several
   publishers can be notified as each finishes. It checks for the required
   entries and returns while any are missing. An explicit null entry lets a
   consumer tell that an upstream branch finished without a result and decide
   whether it can proceed with the remaining inputs. For example, with two
   dependencies, the first callback can return if the second channel is absent.
   Once both channels have entries, the consumer can use the one available
   value even if the other entry is null. A consumer that can be notified
   concurrently must coordinate the check and publication.
4. **Keep the cycle alive while work is pending.** Work queued for later
   captures the context by value so it can read and publish into the same
   cycle. The loop skips a slot while a callback, queued task, or other owner
   retains its context. If all slots are occupied this way, no new cycle
   starts on that scan.
5. **Destroy the context when its last owner releases it.** In normal
   operation, once outside references are gone the loop retains the old context
   until it stores the replacement in that slot. The new cycle’s entry-point
   callbacks therefore run before the old context in that slot is destroyed.
   `Stop()` requests the loop thread to stop, joins it, and clears its slot
   references. Any context still held by a worker survives until that worker
   releases it. If WPILog is configured, the context destructor records its
   available loggable messages before releasing the message map.

## Pipeline

[`second_bot_main.cc`](src/main/second_bot_main.cc) starts NetworkTables for
team 971 and waits for a connection and clock synchronization on normal startup.
It creates a control loop with a 1 ms sleep period and one shared worker pool.
The camera branch runs for each configuration in
[`constants/second_bot/`](constants/second_bot/), including `color.json`.
The front, left, right, and color streams use ports 5801 through 5804,
respectively, all at `/stream`.

```text
Per camera
  UVCCameraNode ──┬── JpegBufferStreamerNode
                  ├── JpegBufferLogNode (optional)
                  └── NvjpegFdDecodeNode
                        └── NvidiaApriltagDetectorNode
                              └── MultiTagSolverNode

All per-camera solver outputs
  UnambiguousSolverNode → VarianceCalculatorNode
                           ├── PositionEstimateRioStreamerNode
                           └── SimulationPositionSenderNode
```

### Per-camera nodes

| Node | Role |
| --- | --- |
| [`UVCCameraNode`](src/camera/uvc_camera_node.cc) | Captures MJPEG frames and timestamps them in roboRIO time. As a dependency node, it puts the latest buffered frame on `jpeg_buffer:<name>` at the start of a cycle, or writes a null entry if no frame is available. |
| [`JpegBufferStreamerNode`](src/streamer/jpeg_buffer_streamer_node.cc) | Serves the original JPEG frames as an HTTP MJPEG stream. |
| [`JpegBufferLogNode`](src/logging/jpeg_buffer_log_node.cc) | Queues writes of the original JPEGs to a per-camera run directory, using the frame timestamp as the filename. This branch is registered when `--log_images` is enabled. The default logging condition uses the Driver Station enabled state from NetworkTables and keeps logging through the autonomous-to-teleop gap. |
| [`NvjpegFdDecodeNode`](src/camera/nvjpeg_fd_decode_node.cc) | Uses hardware JPEG decoding on the shared worker pool and publishes a decoded buffer backed by a file descriptor on `hardware_decoded_image:<name>`. |
| [`NvidiaApriltagDetectorNode`](src/apriltag/nvidia_apriltag_detector_node.cc) | Runs VPI AprilTag detection using the configured PVA or CPU backend and publishes detected tag IDs and image corners on `hardware_apriltag_detections:<name>`. |
| [`MultiTagSolverNode`](src/localization/multi_tag_solver_node.cc) | Registered by `UnambiguousSolverNode::AddCamera()` for each camera. It uses known field tag positions, detected corners, and camera calibration to estimate the robot pose. It solves jointly when multiple tags are accepted and can return two candidate poses for a single tag. The robot enables rejection of small or distant tag detections. If detections are missing or the solve fails, it publishes a null entry so pose fusion can still proceed. |

Here `<name>` comes from the camera JSON’s `name` field. The extrinsics store
translation in meters and `rotation_x`, `rotation_y`, and `rotation_z` in
degrees. The loader converts the rotations to radians.

### Pose fusion and outputs

| Node | Role |
| --- | --- |
| [`UnambiguousSolverNode`](src/localization/unambiguous_solver_node.cc) | Waits for an entry from each camera’s solver, then works with the available candidates. It rejects physically impossible poses, chooses candidates based on agreement across cameras and with the previous estimate, and combines them with inverse-variance weights. It publishes `pose` using the mean of the available camera frame timestamps. |
| [`VarianceCalculatorNode`](src/localization/variance_calculator_node.cc) | Uses the tag count and mean tag distance to compute the variance for the fused pose and publishes `pose_with_variance`. |
| [`PositionEstimateRioStreamerNode`](src/streamer/position_estimate_rio_streamer_node.cc) | Sends `/COS/Pose2d`, `/COS/Pose3d`, and `/COS/PositionEstimate` over NetworkTables. The last topic contains X and Y in meters, yaw in radians, variance, and the mean available camera timestamp in roboRIO seconds. |
| [`SimulationPositionSenderNode`](src/simulation/simulation_position_sender_node.cc) | Sends the pose as JSON with position, quaternion, and timestamp to local viewer clients at `ws://127.0.0.1:5805/robot`. |

### Logging and runtime options

Each run writes `cos.wpilog` under `/cos/logs/logN/`. The log includes pose
publications and available per-camera decode and detection timings. JPEGs, when
recorded, are stored in camera-named subdirectories of the same run.

| Flag | Executable default | Effect |
| --- | --- | --- |
| `--pva_detection` | `true` | Selects the VPI PVA backend. Use `--nopva_detection` for the CPU backend. |
| `--max_context` | `1` | Sets the number of in-flight context slots. |
| `--latency_log` | `false` | Enables control-loop latency and throughput logging. |
| `--log_images` | `true` | Registers the per-camera JPEG logging branches. |

The checked-in [Second Bot service](systemd/second-bot.service) runs
`/root/main/second_bot_main --nopva_detection --max_context=2`.
On shutdown, the executable stops the control loop first, then drains and joins
the worker pool.
