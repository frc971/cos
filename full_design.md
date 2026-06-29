# Full Design

## Overview

COS is a CMake/C++20 vision and localization stack for UVC cameras on NVIDIA
Orin-class hardware. The current runtime is a callback pipeline coordinated by
`control_loops::LoopController`:

1. UVC camera callbacks continuously write the latest JPEG frame for each
   camera into the controller.
2. `LoopController::Run()` snapshots the latest frame and metadata for every
   configured camera into one iteration.
3. Iteration callbacks stream the JPEG directly when streaming is enabled, and
   send the frame through decode, AprilTag detection, multi-camera pose solving,
   and position publishing.
4. A reference-counted `control_loops::Context` keeps the controller from
   starting the next iteration until asynchronous downstream work has released
   its copy.

The implemented wiring entry point is `src/runners/example.cc`, built as the
`localization_example` executable.

---

## Current Module Layout

| Path | Role |
|------|------|
| `src/control_loops/context.h` | Iteration `Context`, `MetaData`, and `MetaDataList`. |
| `src/control_loops/loop_controller.h/.cc` | Latest-frame storage, iteration dispatch, stop/wakeup handling. |
| `src/utils/node.h` | Generic callback interface used by decode, detect, and solver nodes. |
| `src/camera/uvc_camera_node.h/.cc` | UVC MJPEG capture. This node is intentionally not an `INode`. |
| `src/camera/nvjpeg_decode_node.h/.cc` | Asynchronous NVIDIA JPEG decode node. |
| `src/apriltag/apriltag_detector.h` | Detector interface and `NvBufferToGray` helper. |
| `src/apriltag/gpu_apriltag_detector_node.h/.cc` | 971/WPILib GPU AprilTag detector implementation. |
| `src/apriltag/opencv_apriltag_detector_node.h/.cc` | CPU OpenCV AprilTag detector implementation. |
| `src/localization/square_solver_node.h/.cc` | Single-tag ambiguous pose solver. |
| `src/localization/multi_tag_solver_node.h/.cc` | Per-camera multi-tag ambiguous pose solver. |
| `src/localization/unambiguous_solver_node.h/.cc` | Multi-camera accumulator and unambiguous pose selector. |
| `src/localization/position_sender.h` | `IPositionSender` interface. |
| `src/localization/networktable_sender.h/.cc` | NetworkTables publisher for final estimates and debug channels. |
| `src/localization/sim_sender.h` | Header-only in-process sender for tests/simulation. |
| `src/streamer/jpeg_buffer_streamer_node.h/.cc` | Synchronous MJPEG stream sink. |
| `src/logging/png_image_log_node.h/.cc` | PNG image logging node. |
| `src/tools/*.cc` | JPEG log encode/decode/extract tools and UVC logger. |
| `src/runners/example.cc` | Production-style localization pipeline wiring. |
| `src/examples/mjpeg_streamer.cc` | Standalone MJPEG streamer example. |

Top-level CMake includes `src`, `third_party`, and `unit_tests`. The root
`CMakeLists.txt` requires CUDA, JPEG, OpenCV, WPILib, Eigen3, VPI, and libuvc,
sets C++20/CUDA20, and targets CUDA architecture 87.

---

## Iteration Data Flow

```
UVCCameraNode[i]
  -> LoopController::ReceiveFrame(i, jpeg, timestamp)

LoopController::Run(), once per iteration:
  snapshot latest frame + MetaDataList for each camera
  create shared_ptr<control_loops::Context>
  for each camera i:
    call registered iteration callbacks(jpeg, metadata, ctx)
      -> optional JpegBufferStreamerNode::Stream(jpeg)
      -> NvjpegDecodeNode::Decode(jpeg, metadata, ctx)
           async decode thread
             -> detector->Detect(decoded_frame, metadata, ctx)
                  -> solver->Accumulate(detections, metadata, ctx)
                       after all cameras report:
                         SolveWithoutNotify(...)
                         callback(position_estimate_t, output_metadata, ctx)
                           -> IPositionSender::Send(...)
  wait until Context::~Context calls WakeUp(), or RequestStop() is called
```

The controller dispatches callbacks even when a camera has no frame yet. In
that case the JPEG pointer is `nullptr`; `NvjpegDecodeNode` forwards a null
decoded frame and the detectors emit an empty detection vector. This lets
`UnambiguousSolverNode` receive one report per camera per iteration.

---

## Context and Metadata

`src/control_loops/context.h` defines:

```cpp
namespace control_loops {

class LoopController;

struct MetaData {
  int camera_idx = -1;
  unsigned long timestamp = 0;
};

using MetaDataList = std::vector<MetaData>;

struct Context {
  std::atomic<bool> stop_token{false};
  std::weak_ptr<LoopController> controller;
  ~Context();
};

}  // namespace control_loops
```

`Context::~Context()` is implemented in `loop_controller.cc`; it locks the weak
controller pointer and calls `LoopController::WakeUp()` when possible. The
current loop does not read `Context::stop_token`; stopping the runtime is done
through `LoopController::RequestStop()`.

`MetaDataList` is the metadata channel for the pipeline. The controller creates
one metadata entry per camera snapshot:

```cpp
MetaData{.camera_idx = camera_idx, .timestamp = timestamp}
```

The timestamp is the UVC capture timestamp converted to microseconds by
`UVCCameraNode`.

---

## Node Callback Contract

All pipeline nodes except `UVCCameraNode` use the generic interface in
`src/utils/node.h`:

```cpp
template <typename T>
class INode {
 public:
  virtual void RegisterCallback(
      const std::function<void(T, control_loops::MetaDataList,
                               std::shared_ptr<control_loops::Context>)>&
          callback) = 0;
  virtual ~INode() = default;
};
```

Every callback receives:

| Argument | Meaning |
|----------|---------|
| `T` | The node output payload. |
| `control_loops::MetaDataList` | Camera index and capture timestamp metadata. |
| `std::shared_ptr<control_loops::Context>` | Iteration lifetime token. |

`UVCCameraNode` is the exception because camera callbacks run before an
iteration context exists:

```cpp
void RegisterCallback(
    const std::function<void(std::shared_ptr<JpegBuffer>,
                             unsigned long timestamp)>& callback);
```

---

## Loop Controller

`LoopController` owns:

- `latest_frames_`: latest JPEG per camera.
- `latest_metadata_`: latest metadata list per camera.
- `frame_mutexes_`: one mutex per camera slot.
- `callbacks_`: iteration callbacks grouped by camera.
- `wakeup_cv_` and `should_wake_`: the end-of-iteration gate.
- `stop_requested_`: global loop stop flag.

`ReceiveFrame()` overwrites a camera's latest frame and metadata under that
camera's mutex. It does not block on downstream pipeline work.

`Run()` snapshots all camera slots, creates a `Context`, dispatches every
registered callback for every camera, drops its own context reference, and then
waits for `WakeUp()`. If there are no callbacks at all, it calls `WakeUp()`
itself so the loop does not deadlock.

`RequestStop()` sets `stop_requested_` and notifies the condition variable. It
is used by the signal handler in `src/utils/stop.h`.

---

## Camera and Decode

`UVCCameraNode` opens a UVC device based on `UVCCameraConfig`, copies incoming
MJPEG payloads into `JpegBuffer`, converts `uvc_frame_t::capture_time` to an
unsigned-long microsecond timestamp, and calls its registered callbacks.

`NvjpegDecodeNode` implements:

```cpp
INode<std::shared_ptr<camera::DecodedJpegNvBuffer>>
```

`Decode(jpeg, metadata, ctx)` queues work on a `std::jthread`. The worker calls
`DecodeJpegBuffer()`, decodes through `NvJPEGDecoder`, wraps the `NvBuffer` in a
`std::shared_ptr<DecodedJpegNvBuffer>`, and forwards it to registered
callbacks. If the input JPEG is null, it forwards a null decoded frame with the
same metadata and context.

---

## AprilTag Detection

`IApriltagDetectorNode` implements:

```cpp
INode<std::shared_ptr<std::vector<apriltag::tag_detection_t>>>
```

and exposes:

```cpp
virtual void Detect(
    const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
    control_loops::MetaDataList metadata,
    std::shared_ptr<control_loops::Context> ctx) = 0;
```

Both detector implementations preserve metadata and context:

- `GPUApriltagDetectorNode` uses the 971/WPILib GPU detector and the configured
  intrinsics.
- `OpenCVApriltagDetectorNode` uses OpenCV ArUco with the
  `DICT_APRILTAG_36h11` dictionary.

If the decoded frame is null or detection fails, both implementations call
their callbacks with an empty detection vector.

The capture timestamp is not stored in `tag_detection_t`; timestamp and camera
identity travel through `MetaDataList`.

---

## Localization

`position_estimate_t` currently contains:

```cpp
std::vector<int> tag_ids;
std::vector<int> rejected_tag_ids;
wpi::math::Pose3d pose;
double variance;
int num_tags;
double avg_tag_dist;
bool invalid = false;
double loss = 0;
```

There is no timestamp field on `position_estimate_t`; publishers use
`MetaDataList` to attach NetworkTables timestamps.

### Solver Interfaces

`IPositionSolverNode` is the per-camera ambiguous solver interface:

```cpp
INode<ambiguous_estimate_t>
void AmbiguousSolve(detections, metadata, ctx, bool reject_far_tags = true)
```

`SquareSolverNode` and `MultiTagSolverNode` implement this interface. The
unambiguous solver uses their `*WithoutNotify` helpers internally.

`IAccumulatingSolverNode` is the multi-camera final solver interface:

```cpp
INode<position_estimate_t>
void Accumulate(detections, metadata, ctx)
```

### UnambiguousSolverNode

`UnambiguousSolverNode` is constructed with camera constants and an AprilTag
field layout. It does not own a sender. Instead, it publishes final estimates
through registered callbacks.

Current accumulation behavior:

1. `Accumulate()` requires non-empty metadata and reads `camera_idx` from the
   first metadata entry.
2. Each camera is counted once per iteration via `camera_reported_`.
3. For each camera, newer metadata replaces older metadata. Non-empty
   detections replace that camera's accumulated detections only when the
   metadata is newer.
4. When every configured camera has reported, `SolveAndReset()` runs.
5. `SolveAndReset()` calls `SolveWithoutNotify(accumulated_detections_,
   accumulated_metadata_)`, flattens the accumulated metadata into
   `output_metadata`, resets the reported-camera flags, unlocks the mutex, and
   invokes registered callbacks if a result exists.

There is no accumulation timer in the current implementation. The solver waits
for all configured cameras to report once per controller iteration.

`GetAmbiguousEstimates()` uses metadata timestamps to reject stale camera
batches. A camera batch is skipped when its timestamp is at least
`kacceptable_frame_recency` microseconds older than the latest non-empty
detection batch in the same solve.

The final aggregate estimate currently sets `tag_ids`, `pose`, `variance`,
`num_tags`, and `loss`. It does not currently aggregate `rejected_tag_ids` or
`avg_tag_dist` from the per-camera estimates.

---

## Position Sending

`IPositionSender` is not an `INode`; it is the final sink interface:

```cpp
virtual void Send(const position_estimate_t& estimate,
                  control_loops::MetaDataList metadata,
                  std::shared_ptr<control_loops::Context> ctx) = 0;
```

`NetworkTableSender` publishes to:

```text
Orin/PoseEstimate/<sender_name>
```

It publishes:

- `Pose` as `wpi::math::Pose2d`
- `Pose3d` as `wpi::math::Pose3d`
- `TagEstimation` as `[x, y, yaw, variance, num_tags, avg_tag_dist, loss]`
- `TagId` and `RejectedTagId` as boolean arrays of length 50
- `NumTags`, `Variance`, `AvgTagDist`, and `Loss`

The NetworkTables timestamp is the maximum timestamp found in the output
metadata. `NetworkTableSender` also exposes `MakeDoubleChannel`,
`MakeBoolChannel`, and `MakeStringChannel` helpers for ad hoc debug publishers.

`SimSender` stores `last_estimate`, `last_metadata`, and `last_context` in
process. It is used by tests and by `localization_example --sim_sender`.

---

## Runtime Wiring in `src/runners/example.cc`

`localization_example` performs this setup:

1. Parse Abseil flags:
   - `--sim_sender`
   - `--sender_name`, default `localization`
2. Load camera constants with `camera::GetCameraConstants()`.
3. Keep only UVC cameras and sort them by camera name.
4. Create either `SimSender` or `NetworkTableSender`.
5. Create `UnambiguousSolverNode` and register a callback that calls
   `sender->Send(estimate, metadata, ctx)`.
6. Create `LoopController` with the number of configured UVC cameras.
7. For each camera:
   - Create `UVCCameraNode`.
   - Register its callback to call `controller->ReceiveFrame(...)`.
   - If a stream port is configured, create `JpegBufferStreamerNode` and
     register a controller iteration callback for direct JPEG streaming.
   - Create `NvjpegDecodeNode`.
   - Create `GPUApriltagDetectorNode` for `austin_gpu` or
     `OpenCVApriltagDetectorNode` for `opencv_cpu`.
   - Wire decoder callback to detector `Detect()`.
   - Wire detector callback to solver `Accumulate()`.
   - Register a controller iteration callback that calls decoder `Decode()`.
8. Register signal handling with `stop::RegisterHandler([controller] {
   controller->RequestStop();
   })`.
9. Start all cameras, then call `controller->Run()`.

Configured UVC camera constants live in `constants/camera_constants.json`.
The current checked-in configuration defines three old-first-bot UVC cameras
using the `austin_gpu` detector and ports `5801`, `5802`, and `5803`.

---

## Timestamp Handling

The authoritative frame timestamp is the UVC capture time converted to
microseconds:

```text
UVCCameraNode::CallBack()
  -> LoopController::ReceiveFrame(camera_idx, jpeg, timestamp)
  -> latest_metadata_[camera_idx] = {MetaData{camera_idx, timestamp}}
  -> iteration callback(jpeg, metadata, ctx)
  -> NvjpegDecodeNode::Decode(jpeg, metadata, ctx)
  -> detector->Detect(decoded, metadata, ctx)
  -> solver->Accumulate(detections, metadata, ctx)
  -> solver callback(position_estimate_t, output_metadata, ctx)
  -> IPositionSender::Send(estimate, output_metadata, ctx)
  -> NetworkTables Set(..., max(metadata.timestamp))
```

The timestamp is not re-derived from decode, detection, solve, or publish time.

---

## Build Targets and Tests

Primary source subdirectories:

- `src/control_loops`
- `src/utils`
- `src/apriltag`
- `src/camera`
- `src/streamer`
- `src/localization`
- `src/logging`
- `src/tools`
- `src/runners`
- `src/examples`

Notable executables:

- `localization_example`
- `uvc_logger`
- `jpeg_extract_log`
- `jpeg_decode_frames`
- `jpeg_encode_frames`
- `mjpeg_streamer`

Current unit test areas:

- AprilTag detector nodes
- camera constants
- UVC camera node
- NVJPEG decode node
- loop controller
- NetworkTables sender
- position sender
- position solver and solver nodes
- JPEG buffer streamer
- JPEG frame tools

Build commands from `README.md`:

```sh
cmake -B build -G Ninja -Wno-dev
cmake --build build
```

---

## Extension Points

- Add another camera source by feeding `LoopController::ReceiveFrame()` with a
  JPEG buffer and capture timestamp.
- Add another decoder by implementing `INode<decoded-frame-type>` and preserving
  `MetaDataList` plus `Context` through callbacks.
- Swap detector implementations through `IApriltagDetectorNode`.
- Add debug logging by registering additional callbacks on existing nodes or by
  using `NetworkTableSender::Make*Channel`.
- Swap final output by implementing `IPositionSender` and changing the sender
  constructed in `src/runners/example.cc`.
