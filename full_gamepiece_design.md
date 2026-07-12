# Gamepiece Detection — Full Implementation Design

## Overview

This document describes the implementation plan for gamepiece detection in COS.
It adds a second, independent control loop that taps the existing localization
decode nodes to run YOLO inference on each camera frame. The gamepiece loop runs
on its own cycle time and has no effect on the localization loop's timing.

Reference: `gamepiece_design_draft.md` and the bos implementations in
`../bos/src/gamepiece/` and `../bos/src/yolo/`.

---

## Goals

- Minimal holdup of the main localization loop.
- Interacts with the existing pipeline only through the NvjpegDecodeNodes
  already owned by localization.
- Maintains a separate GamepieceLoopController with its own cycle time.
- YOLO nodes are fully node-styled and callback-wired, but no post-YOLO
  callbacks are registered at this time (placeholders for future use).

---

## Architecture

```
                 Localization Loop (existing, unchanged)
UVCCameraNode[i]
  -> LocalizationLoopController::ReceiveFrame(i, jpeg, timestamp)
  -> NvjpegDecodeNode::Decode(jpeg, metadata, ctx)
       |
       +----> [existing] ApriltagDetectorNode -> ... -> NetworkTableSender
       |
       +----> [NEW] GamepieceLoopController::ReceiveFrame(i, decoded, timestamp)
                    (woken on each decoded frame from each camera)

                 Gamepiece Loop (new)
GamepieceLoopController::Run()
  snapshot latest DecodedJpegNvBuffer per camera
  create shared_ptr<Context>
  for each gamepiece camera i:
    -> YoloNode::Detect(decoded, metadata, ctx)  [own worker thread]
         NvBuffer -> cv::Mat conversion
         YOLO preprocess + TRT inference + postprocess
         Pinhole geometry -> robot-relative Pose3d per detection
         -> callback(vector<gamepiece_detection_t>, metadata, ctx)
              [no callbacks registered yet — placeholder]
  wait for Context::~Context -> WakeUp()
```

---

## Refactor: `LoopController<SyncedObjType>` Template

### Motivation

The existing `LoopController` is specialized for `shared_ptr<JpegBuffer>`.
The gamepiece loop needs the identical coordination logic but over
`shared_ptr<camera::DecodedJpegNvBuffer>`. Rather than duplicate the class,
`LoopController` becomes a template.

### Changes to `src/control_loops/loop_controller.h`

```cpp
template <typename SyncedObjType>
class LoopController : public std::enable_shared_from_this<LoopController<SyncedObjType>> {
 public:
  explicit LoopController(int num_cameras);

  void ReceiveFrame(int camera_idx,
                    std::shared_ptr<SyncedObjType> frame,
                    unsigned long timestamp);

  void RegisterIterationCallback(
      int camera_idx,
      std::function<void(std::shared_ptr<SyncedObjType>,
                         MetaDataList,
                         std::shared_ptr<Context>)> callback);

  void Run();
  void WakeUp();
  void RequestStop();

 private:
  // identical to current implementation, frame type is SyncedObjType
  int num_cameras_;
  std::vector<std::shared_ptr<SyncedObjType>> latest_frames_;
  std::vector<MetaDataList> latest_metadata_;
  std::vector<std::mutex> frame_mutexes_;
  std::vector<std::vector<
      std::function<void(std::shared_ptr<SyncedObjType>,
                         MetaDataList,
                         std::shared_ptr<Context>)>>> callbacks_;
  std::mutex wakeup_mutex_;
  std::condition_variable wakeup_cv_;
  bool should_wake_{false};
  std::atomic<bool> stop_requested_{false};
};

// Convenience aliases
using LocalizationLoopController = LoopController<camera::JpegBuffer>;
using GamepieceLoopController    = LoopController<camera::DecodedJpegNvBuffer>;
```

### Migration

- Move the full template implementation into `loop_controller.h` (templates
  must be header-only).
- `loop_controller.cc` retains only `Context::~Context`.
- All existing call sites use `LocalizationLoopController` or continue to use
  `LoopController<camera::JpegBuffer>` directly — both name the same type.
- The public API of each instantiation is identical to the current
  `LoopController` API, so `example.cc` call sites need only a type alias
  change.

---

## Camera Constants Additions

Two new optional fields are added to `camera_constant_t`
(`src/camera/camera_constants.h`):

```cpp
std::optional<std::string> yolo_model_path = std::nullopt;
// If present, this camera participates in the gamepiece loop and uses this
// TRT engine file path.
bool run_gamepiece = false;
```

Parsing is added to `camera_constants.cc` alongside the existing fields.

A camera participates in the gamepiece loop if and only if
`run_gamepiece == true`. The `yolo_model_path` must be present when
`run_gamepiece` is true; the runner CHECK-fails otherwise.

Example `camera_constants.json` entry addition:

```json
{
  "run_gamepiece": true,
  "yolo_model_path": "/path/to/yolo_gamepiece.engine"
}
```

Class names (e.g. `["coral", "algae"]`) are passed as a flag in the runner
(see Runner section below).

---

## Output Type: `gamepiece_detection_t`

New file: `src/gamepiece/gamepiece_detection.h`

```cpp
#pragma once
#include <frc/geometry/Pose3d.h>
#include <string>

namespace gamepiece {

struct gamepiece_detection_t {
  frc::Pose3d pose;       // Pose3d of the gamepiece relative to the robot
  int tracker_id = -1;    // -1 until tracking is implemented
  int class_id;           // Index into class_names (e.g. 0=coral, 1=algae)
  float confidence;
};

}  // namespace gamepiece
```

The `tracker_id` field is reserved for future tracking integration. For now
every detection gets `-1`.

---

## `YoloNode`

New files: `src/gamepiece/yolo_node.h`, `src/gamepiece/yolo_node.cc`

### Interface

```cpp
namespace gamepiece {

class YoloNode : public INode<std::vector<gamepiece_detection_t>> {
 public:
  YoloNode(const std::string& model_path,
           const std::vector<std::string>& class_names,
           nlohmann::json intrinsics,
           nlohmann::json extrinsics);
  ~YoloNode() override;

  void RegisterCallback(
      const std::function<void(std::vector<gamepiece_detection_t>,
                               control_loops::MetaDataList,
                               std::shared_ptr<control_loops::Context>)>&
          callback) override;

  void Detect(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx);

 private:
  void RunDetection(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
                    control_loops::MetaDataList metadata,
                    std::shared_ptr<control_loops::Context> ctx);

  cv::Mat NvBufferToMat(NvBuffer* nv_buffer);
  auto ComputePose(const cv::Rect& bbox) const -> frc::Pose3d;

  yolo::Yolo yolo_;
  const std::vector<std::string> class_names_;

  // Intrinsics (from camera_constants.json)
  float cam_cx_, cam_cy_, fx_, fy_;

  // Extrinsics (from camera_constants.json)
  float cam_pitch_;
  float pinhole_height_;    // translation_z (camera height above ground)
  frc::Pose3d cam_pose_;    // full camera pose for TransformBy

  std::vector<std::function<void(std::vector<gamepiece_detection_t>,
                                 control_loops::MetaDataList,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;

  std::mutex mutex_;
  std::condition_variable_any cv_;
  std::queue<std::function<void()>> tasks_;
  std::jthread worker_thread_;
};

}  // namespace gamepiece
```

### Threading

Mirrors `NvjpegDecodeNode`: `Detect()` enqueues a task; the `worker_thread_`
(a `std::jthread`) dequeues and calls `RunDetection()`.

### NvBuffer → cv::Mat Conversion

`NvBufferToMat()` converts an `NvBuffer` (from `DecodedJpegNvBuffer`) to a
`cv::Mat`. The decode node produces a planar buffer; the function maps the Y
plane for grayscale YOLO (or interleaves planes for color). The `yolo::Yolo`
constructor `color` flag is set accordingly and matches what the TRT engine
expects.

Implementation note: use `NvBufferMemMap` / `NvBufferMemUnMap` to access
buffer planes CPU-side (same pattern used in other Tegra repos). The mapped
plane data is copied into a `cv::Mat` before unmapping.

### Pose Computation

Ported directly from `../bos/src/gamepiece/gamepiece.cc`:

```
pinhole_height  = extrinsics["translation_z"]   (camera height above ground)
cam_cx, cam_cy  = intrinsics["cx"], intrinsics["cy"]
fx, fy          = intrinsics["fx"], intrinsics["fy"]
cam_pitch       = extrinsics["rotation_y"]
cam_pose        = Pose3d from full extrinsics translation + rotation

For each detection bbox:
  c_y = bbox.y + bbox.height/2
  c_x = bbox.x + bbox.width/2
  cam_relative_pitch = atan2(c_y - cam_cy, fy)
  phi = cam_relative_pitch + cam_pitch
  distance = pinhole_height / sin(phi)
  cam_relative_yaw = atan2(c_x - cam_cx, fx)

  target_pose_cam_relative = Transform3d(
    Translation3d(
      distance * cos(cam_relative_pitch) * cos(cam_relative_yaw),
      distance * cos(cam_relative_pitch) * sin(cam_relative_yaw),
      distance * -sin(cam_relative_pitch)
    ),
    Rotation3d(0, cam_relative_pitch, -cam_relative_yaw)
  )

  target_pose_robot_relative = cam_pose.TransformBy(target_pose_cam_relative)
```

The result is stored as `gamepiece_detection_t::pose`.

### Output

`RunDetection()` calls `yolo_.RunModel(mat)` and `yolo_.Postprocess(...)`,
then iterates `MAX_DETECTIONS = 6` results, computes `gamepiece_detection_t`
for each non-empty bbox, and fires registered callbacks with the full vector.
If the input frame is null, callbacks are called with an empty vector.

---

## Wiring in `src/runners/example.cc`

### New flags

```cpp
ABSL_FLAG(bool, run_gamepiece, false,
          "Enable the gamepiece YOLO detection loop");
ABSL_FLAG(std::vector<std::string>, gamepiece_class_names,
          {"coral", "algae"},
          "Ordered class names matching the YOLO model output");
```

### Setup sequence (additions after existing localization wiring)

```cpp
if (absl::GetFlag(FLAGS_run_gamepiece)) {
  // Collect gamepiece-enabled cameras in the same sorted order.
  std::vector<int> gamepiece_camera_indices;
  for (size_t i = 0; i < camera_constants.size(); ++i) {
    if (camera_constants[i].run_gamepiece) {
      gamepiece_camera_indices.push_back(static_cast<int>(i));
    }
  }
  CHECK(!gamepiece_camera_indices.empty())
      << "run_gamepiece is set but no cameras have run_gamepiece=true";

  auto gamepiece_controller =
      std::make_shared<control_loops::GamepieceLoopController>(
          camera_constants.size());  // same slot count as localization

  std::vector<std::string> class_names =
      absl::GetFlag(FLAGS_gamepiece_class_names);

  std::vector<std::unique_ptr<gamepiece::YoloNode>> yolo_nodes;

  for (int camera_idx : gamepiece_camera_indices) {
    const auto& constant = camera_constants[camera_idx];
    CHECK(constant.yolo_model_path.has_value())
        << "Camera " << constant.name << " has run_gamepiece=true "
        << "but is missing yolo_model_path";
    CHECK(constant.intrinsics_path.has_value());
    CHECK(constant.extrinsics_path.has_value());

    const nlohmann::json intrinsics = ReadJsonFile(*constant.intrinsics_path);
    const nlohmann::json extrinsics = ReadJsonFile(*constant.extrinsics_path);

    auto yolo_node = std::make_unique<gamepiece::YoloNode>(
        *constant.yolo_model_path, class_names, intrinsics, extrinsics);

    // No callbacks registered on yolo_node yet (placeholder for future use).

    auto* yolo_node_ptr = yolo_node.get();
    auto* gamepiece_ctrl_ptr = gamepiece_controller.get();

    // Tap the existing decode node for this camera.
    // Each decoded frame wakes the gamepiece loop.
    decoders[camera_idx]->RegisterCallback(
        [gamepiece_ctrl_ptr, camera_idx](
            const std::shared_ptr<camera::DecodedJpegNvBuffer>& decoded,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> /*loc_ctx*/) {
          // Extract timestamp from metadata (first entry, same pattern as localization).
          unsigned long ts = metadata.empty() ? 0 : metadata[0].timestamp;
          gamepiece_ctrl_ptr->ReceiveFrame(camera_idx, decoded, ts);
        });

    // Register a gamepiece iteration callback that runs YOLO for this camera.
    gamepiece_controller->RegisterIterationCallback(
        camera_idx,
        [yolo_node_ptr](
            const std::shared_ptr<camera::DecodedJpegNvBuffer>& decoded,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> ctx) {
          yolo_node_ptr->Detect(decoded, std::move(metadata), std::move(ctx));
        });

    yolo_nodes.push_back(std::move(yolo_node));
  }

  // Run the gamepiece loop on a separate thread so it does not block
  // the localization loop.
  std::thread gamepiece_thread(
      [gamepiece_controller] { gamepiece_controller->Run(); });
  gamepiece_thread.detach();
  // stop::RegisterHandler already handles RequestStop for the localization
  // controller; add gamepiece controller stop here.
}
```

The `stop::RegisterHandler` call is updated to also stop the gamepiece
controller when a signal is received.

---

## File Layout

```
src/
  control_loops/
    context.h             (unchanged)
    loop_controller.h     (refactored to template; full impl moves here)
    loop_controller.cc    (retains only Context::~Context)
  camera/
    camera_constants.h    (add yolo_model_path, run_gamepiece fields)
    camera_constants.cc   (parse new fields from JSON)
  gamepiece/
    CMakeLists.txt
    gamepiece_detection.h (gamepiece_detection_t struct)
    yolo_node.h
    yolo_node.cc
  runners/
    example.cc            (new flags + gamepiece wiring block)
```

No `src/yolo/` directory. The bos `yolo::Yolo` class is referenced as a
dependency (via CMake subdir or copy — to be decided during implementation;
mirror the approach used for other bos utilities already in third_party or
directly include).

---

## CMake

`src/gamepiece/CMakeLists.txt`:

```cmake
add_library(gamepiece
  yolo_node.cc
)
target_link_libraries(gamepiece
  PUBLIC
    camera_nodes        # for DecodedJpegNvBuffer
    control_loops
    yolo                # bos yolo library (or cos-local copy)
    wpilib_geometry
    nlohmann_json
    opencv_core
    opencv_cudawarping
    cuda_runtime
    nvbuf_utils         # for NvBufferMemMap
)
```

`src/CMakeLists.txt`: add `add_subdirectory(gamepiece)`.

`src/runners/CMakeLists.txt`: link `localization_example` against `gamepiece`.

---

## Unit Tests

The following tests should be added in a future step (not part of this initial
implementation pass):

| Test file | What it covers |
|-----------|---------------|
| `unit_tests/gamepiece/yolo_node_test.cc` | YoloNode with a synthetic decoded buffer (null frame → empty detections, non-null frame → callback fires). |
| `unit_tests/gamepiece/gamepiece_detection_test.cc` | Pose computation math against known inputs/outputs from the bos reference. |

---

## Open Items / Future Work

- **Tracking**: `tracker_id` is `-1` in all detections. A tracker (e.g. SORT or
  IoU-based) will be added as a post-YOLO node when the NT publishing design is
  finalized.
- **Post-YOLO callbacks**: NT publishing of `gamepiece_detection_t` (Pose3d +
  tracker_id + class_id as a custom struct) will be added when the output
  layout is decided.
- **NvBuffer memory layout**: The exact plane format of `DecodedJpegNvBuffer`
  depends on the NvJPEG decoder output format. `NvBufferToMat` in `YoloNode`
  must handle the correct NvBufSurfaceColorFormat (likely NvBufSurfaceColorFormat_GRAY8
  or NvBufSurfaceColorFormat_YUV420). Verify against the decode node output
  during implementation.
- **YOLO color mode**: Whether the TRT engine is grayscale or color must match
  the `yolo::Yolo` constructor `color` flag. Add a `yolo_color` boolean to
  camera_constants.json if cameras need different modes.
