# Full Design Plan

## Overview

This document specifies the implementation plan for the refactored localization system. It maps every file, class, method, and data flow that needs to be created or changed.

The core architectural shift from `/bos` is replacing the blocking-loop-per-camera-thread model with a single `LoopController` that manages iteration lifecycle via a reference-counted `Context` object. Nodes are wired via callbacks and the iteration gate is implicit: when all nodes release their reference to the current `Context`, the destructor wakes the controller to begin the next iteration.

---

## Data Flow (one iteration, two cameras)

```
UVCCameraNode[0] ──writes──► LoopController.latest_frames_[0]
UVCCameraNode[1] ──writes──► LoopController.latest_frames_[1]

LoopController.Run() per iteration:
  copy latest_frames_ → per_iter_frames_[0..N]
  create shared_ptr<Context> ctx
  for each camera i:
    call registered callbacks(per_iter_frames_[i], ctx)
      ├─ JpegBufferStreamerNode::Stream(jpeg)        [no ctx needed; synchronous]
      └─ NvjpegDecodeNode::Decode(jpeg, ctx)
           └─ (async decode thread)
                calls IApriltagDetectorNode::Detect(frame, timestamp, ctx)
                  └─ (detect; may run on GPU)
                       calls solver callback(detections, camera_idx=i, ctx)
                         └─ UnambiguousSolverNode::Accumulate(...)
                              when all N cameras reported:
                                Solve() → IPositionSender::Send(estimate)
                              release ctx reference ──► ctx destructor ──► WakeUp()
  wait on condition_variable
```

---

## File Layout

### New files

| Path | Purpose |
|------|---------|
| `src/utils/context.h` | `Context` struct |
| `src/utils/loop_controller.h` | `LoopController` declaration |
| `src/utils/loop_controller.cc` | `LoopController` implementation |
| `src/localization/position_sender.h` | `IPositionSender` interface |
| `src/localization/networktable_sender.h` | `NetworkTableSender` (from bos, adapted) |
| `src/localization/networktable_sender.cc` | implementation |
| `src/localization/sim_sender.h` | `SimSender` (test/sim stub) |
| `src/main/main.cc` | wiring entry point |

### Modified files

| Path | Change |
|------|--------|
| `src/camera/nvjpeg_decode_node.h/.cc` | `Decode()` gains `shared_ptr<Context>` param; output callback gains it too |
| `src/apriltag/apriltag_detector.h` | `Detect()` gains `shared_ptr<Context>` param; output callback gains it |
| `src/apriltag/opencv_apriltag_detector_node.h/.cc` | same |
| `src/apriltag/gpu_apriltag_detector_node.h/.cc` | same |
| `src/localization/position_solver.h` | replace `IJointPositionSolverNode` with new per-camera accumulator interface |
| `src/localization/unambiguous_solver_node.h/.cc` | rewrite input path; holds sender |
| `src/utils/node.h` | add `shared_ptr<Context>` as second arg to every callback |
| `src/camera/uvc_camera_node.h/.cc` | **exception**: does not implement `INode<T>`; keeps context-free callback |
| `src/streamer/jpeg_buffer_streamer_node.h/.cc` | unchanged |

---

## `src/utils/context.h` (new)

```cpp
#pragma once
#include <atomic>
#include <memory>

namespace utils {

class LoopController;  // forward decl

struct Context {
  std::atomic<bool> stop_token{false};
  std::weak_ptr<LoopController> controller;
  ~Context();
};

}  // namespace utils
```

- `stop_token`: any node can set this to `true` to prevent the next iteration.
- `~Context()` calls `controller.lock()->WakeUp()` if the controller is still alive.
- The `Context` destructor is defined in `loop_controller.cc` (after `LoopController` is complete) to avoid circular include issues — alternatively use a forward-declared impl in context.cc.

---

## `src/utils/loop_controller.h` (new)

```cpp
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "camera/uvc_camera_node.h"   // for JpegBuffer
#include "utils/context.h"

namespace utils {

class LoopController : public std::enable_shared_from_this<LoopController> {
 public:
  explicit LoopController(int num_cameras);

  // Called by UVCCameraNode's registered callback to deliver a new frame.
  // timestamp is the jpeg capture time from uvc_frame_t->capture_time in microseconds.
  // Thread-safe; may be called from camera thread.
  void ReceiveFrame(int camera_idx,
                    std::shared_ptr<camera::JpegBuffer> frame,
                    unsigned long timestamp);

  // Register a callback to be called once per iteration for a given camera.
  // Signature: void(shared_ptr<JpegBuffer>, unsigned long timestamp, shared_ptr<Context>)
  // timestamp is the jpeg capture time snapshotted at ReceiveFrame time.
  // Multiple callbacks per camera are allowed (e.g. streamer + decoder).
  // Must be called before Run().
  void RegisterIterationCallback(
      int camera_idx,
      std::function<void(std::shared_ptr<camera::JpegBuffer>,
                         unsigned long timestamp,
                         std::shared_ptr<Context>)>
          callback);

  // Main loop. Blocks until RequestStop() is called.
  void Run();

  // Called by Context::~Context() to signal iteration completion.
  void WakeUp();

  // Called by signal handler to stop the loop after the current iteration.
  void RequestStop();

 private:
  int num_cameras_;

  // Latest frame and capture timestamp per camera, written by ReceiveFrame().
  std::vector<std::shared_ptr<camera::JpegBuffer>> latest_frames_;
  std::vector<unsigned long> latest_timestamps_;  // parallel to latest_frames_
  std::vector<std::mutex> frame_mutexes_;  // one per camera

  // Downstream callbacks per camera. Index: [camera_idx][callback_idx].
  std::vector<std::vector<std::function<void(std::shared_ptr<camera::JpegBuffer>,
                                             unsigned long timestamp,
                                             std::shared_ptr<Context>)>>>
      callbacks_;

  // Condition variable for end-of-iteration wakeup.
  std::mutex wakeup_mutex_;
  std::condition_variable wakeup_cv_;
  bool should_wake_{false};

  std::atomic<bool> stop_requested_{false};
};

}  // namespace utils
```

---

## `src/utils/loop_controller.cc` (new)

### `Context::~Context()` — defined here after LoopController is complete

```cpp
Context::~Context() {
  if (auto ctrl = controller.lock()) {
    ctrl->WakeUp();
  }
}
```

### `LoopController::LoopController(int num_cameras)`

- Resize `latest_frames_`, `frame_mutexes_`, and `callbacks_` to `num_cameras`.
- `latest_frames_` is initialized to `nullptr`.

### `LoopController::ReceiveFrame(int camera_idx, shared_ptr<JpegBuffer> frame, unsigned long timestamp)`

```cpp
lock frame_mutexes_[camera_idx]
latest_frames_[camera_idx]     = frame      // overwrite; no copy of pixel data
latest_timestamps_[camera_idx] = timestamp  // jpeg capture time
unlock
```

This is the "copy whatever is newest" policy. No blocking.

### `LoopController::Run()`

```cpp
while (!stop_requested_) {
  // Step 1: snapshot latest frames and timestamps (brief lock per camera)
  per_iter_frames[N]
  per_iter_timestamps[N]
  for i in 0..num_cameras_:
    lock frame_mutexes_[i]
    per_iter_frames[i]     = latest_frames_[i]      // shared_ptr copy; zero-copy of data
    per_iter_timestamps[i] = latest_timestamps_[i]  // jpeg capture time
    unlock

  // Step 2: create iteration context
  auto ctx = make_shared<Context>()
  ctx->controller = weak_from_this()

  // Step 3: call all downstream callbacks
  for i in 0..num_cameras_:
    if per_iter_frames[i] == nullptr: continue
    for each cb in callbacks_[i]:
      cb(per_iter_frames[i], per_iter_timestamps[i], ctx)

  // Step 4: release our reference to ctx
  ctx.reset()
  // ctx will destruct when all downstream nodes release their copies

  // Step 5: wait for WakeUp() (which Context::~Context calls)
  unique_lock lock(wakeup_mutex_)
  wakeup_cv_.wait(lock, [this]{ return should_wake_ || stop_requested_; })
  should_wake_ = false

  // Step 6: if any node set ctx.stop_token (now dead), check system stop
  // Note: per-iteration stop_token is checked via stop_requested_ set by RequestStop()
}
```

Note: `ctx.stop_token` is not checked here because the context is already dead when WakeUp fires. Nodes that want to stop the loop should call `controller->RequestStop()` via the weak_ptr — or the design can be extended to pass the stop decision through Context and have WakeUp read a flag. For now, `RequestStop()` is the mechanism.

### `LoopController::WakeUp()`

```cpp
{
  lock_guard lock(wakeup_mutex_)
  should_wake_ = true
}
wakeup_cv_.notify_one()
```

### `LoopController::RequestStop()`

```cpp
stop_requested_ = true
wakeup_cv_.notify_one()  // unblock wait if sleeping
```

---

## `src/utils/node.h` — modified

```cpp
#pragma once
#include <functional>
#include <memory>

namespace utils { struct Context; }

template <typename T>
class INode {
 public:
  virtual void RegisterCallback(
      const std::function<void(T, std::shared_ptr<utils::Context>)>& callback) = 0;
  virtual ~INode() = default;
};
```

Every node in the pipeline (decode, detect, solve) implements `INode<T>` with this two-argument callback. The context is always threaded through.

---

## `src/camera/uvc_camera_node.h/.cc` — exception to INode<T>

`UVCCameraNode` does **not** implement `INode<T>`. The context does not exist yet when the camera fires — it is created by `LoopController` at the start of each iteration, after the frame has been written to the latest-frame slot.

`UVCCameraNode` also carries the jpeg capture timestamp from the UVC driver (`uvc_frame_t->capture_time`). This is the authoritative frame timestamp that flows through the entire pipeline.

```cpp
void RegisterCallback(
    const std::function<void(std::shared_ptr<JpegBuffer>, unsigned long timestamp)>& callback);
```

`CallBack()` extracts `frame->capture_time` (converted to microseconds as an unsigned long) and passes it alongside the buffer.

Wired in `main.cc`:

```cpp
camera.RegisterCallback([&controller, idx](std::shared_ptr<camera::JpegBuffer> frame,
                                           unsigned long timestamp) {
  controller->ReceiveFrame(idx, std::move(frame), timestamp);
});
```

No other changes to `UVCCameraNode`.

---

## `src/camera/nvjpeg_decode_node.h` — modified

Introduces `TimestampedDecodedFrame` to carry the jpeg capture timestamp alongside the decoded buffer through the pipeline. Implements `INode<TimestampedDecodedFrame>`.

```cpp
struct TimestampedDecodedFrame {
  std::shared_ptr<DecodedJpegNvBuffer> buffer;
  unsigned long timestamp;  // jpeg capture time in microseconds, sourced from uvc_frame_t->capture_time
};

class NvjpegDecodeNode : public INode<TimestampedDecodedFrame> {
 public:
  explicit NvjpegDecodeNode(const std::string& name);
  ~NvjpegDecodeNode() override;

  // INode<TimestampedDecodedFrame>
  void RegisterCallback(
      const std::function<void(TimestampedDecodedFrame,
                               std::shared_ptr<utils::Context>)>& callback) override;

  // Input: called by LoopController's iteration callback.
  void Decode(const std::shared_ptr<camera::JpegBuffer>& jpeg_buffer,
              unsigned long timestamp,
              std::shared_ptr<utils::Context> ctx);

 private:
  void DecodeJpegBuffer(const std::shared_ptr<camera::JpegBuffer>& jpeg_buffer,
                        unsigned long timestamp,
                        std::shared_ptr<utils::Context> ctx);

  NvJPEGDecoder* decoder_ = nullptr;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(TimestampedDecodedFrame,
                                 std::shared_ptr<utils::Context>)>>
      callbacks_;
  std::jthread decode_thread_;
};
```

**`Decode()`:** queues a lambda capturing `jpeg_buffer`, `timestamp`, and `ctx` by value.

**`DecodeJpegBuffer()`:** decodes the buffer, then calls each registered callback with `(TimestampedDecodedFrame{buffer, timestamp}, ctx)`.

---

## `src/apriltag/apriltag_detector.h` — modified

`IApriltagDetectorNode` now inherits `INode<std::shared_ptr<std::vector<tag_detection_t>>>`.

```cpp
class IApriltagDetectorNode
    : public INode<std::shared_ptr<std::vector<tag_detection_t>>> {
 public:
  // INode<shared_ptr<vector<tag_detection_t>>>
  virtual void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>,
                               std::shared_ptr<utils::Context>)>& callback) override = 0;

  // Input: called by the decode node's registered callback.
  // timestamp comes from TimestampedDecodedFrame and is the jpeg capture time.
  virtual void Detect(const camera::DecodedJpegNvBuffer& frame,
                      unsigned long timestamp,
                      std::shared_ptr<utils::Context> ctx) = 0;

  virtual ~IApriltagDetectorNode() = default;
};
```

`NvBufferToGray` helper stays here.

---

## `src/apriltag/opencv_apriltag_detector_node.h/.cc` — modified

Implements `IApriltagDetectorNode`. Changes:
- `Detect()` gains `std::shared_ptr<utils::Context> ctx` parameter (satisfying the interface).
- `RegisterCallback` stores `std::function<void(shared_ptr<vector<tag_detection_t>>, shared_ptr<Context>)>`.
- Callbacks called with `(detections, ctx)`. Because `Detect` is synchronous, `ctx` is held on the call stack until all callbacks return, then released naturally.

---

## `src/apriltag/gpu_apriltag_detector_node.h/.cc` — modified

Same changes as `OpenCVApriltagDetectorNode` — implement updated `IApriltagDetectorNode`, propagate `ctx` through `Detect()` to callbacks.

---

## `src/localization/position_sender.h` (new)

```cpp
#pragma once
#include "localization/position.h"

namespace localization {

class IPositionSender {
 public:
  virtual void Send(const position_estimate_t& estimate) = 0;
  virtual ~IPositionSender() = default;
};

}  // namespace localization
```

This replaces the bos `IPositionSender`. Not a node — nothing registers callbacks on senders.

---

## `src/localization/networktable_sender.h/.cc` (new)

Port of `bos/src/localization/networktable_sender.h/.cc` implementing `IPositionSender`.

### Logging via NetworkTables

NetworkTables publishing **is** the logging mechanism. WPILib automatically records all NT publisher values to the wpilog file via `frc::DataLogManager`. There is no separate log path or `DataLogWriter` to wire up.

**`position_estimate_t` is never extended for logging purposes.** Debug values are published directly to NT via callbacks registered on the sender, independent of `Send()`.

`NetworkTableSender` exposes factory methods that create a typed NT publisher and return a `std::function` that, when called with a value, publishes it immediately:

```cpp
std::function<void(double)>      MakeDoubleChannel(const std::string& subkey);
std::function<void(bool)>        MakeBoolChannel(const std::string& subkey);
std::function<void(std::string)> MakeStringChannel(const std::string& subkey);
// Add more overloads as needed
```

Each returned callback is an independent NT publisher scoped under the sender's camera subtable. The caller (typically `main.cc`) holds the callback and passes it to whichever node wants to log that value. The node calls it directly — no `ctx` or `position_estimate_t` involved.

**Adding a new logged field requires touching at most 3 files:**
- *For a type already supported* (e.g. `double`): only `main.cc` — call `MakeDoubleChannel`, hand the callback to the node.
- *For a brand-new NT type*: add `Make*Channel` to `networktable_sender.h` and `networktable_sender.cc`, then wire in `main.cc` — 3 files.

### Header

```cpp
NetworkTableSender(const std::string& camera_name, bool verbose = false);
void Send(const position_estimate_t& estimate) override;

std::function<void(double)>      MakeDoubleChannel(const std::string& subkey);
std::function<void(bool)>        MakeBoolChannel(const std::string& subkey);
std::function<void(std::string)> MakeStringChannel(const std::string& subkey);
```

No `sim` flag — simulation is handled by `SimSender`. No `DataLogWriter` arg — NT logging is automatic.

### Implementation

Identical to bos `networktable_sender.cc` except:
- Uses `cos` includes (`localization/position.h` not `src/localization/position.h`).
- No `latency` field on `position_estimate_t` (field was on bos; check if it exists in cos — if not, omit).
- Uses `absl/log` (not `LOG` from frc).
- No `DataLogWriter` integration — removed entirely.

Each `Make*Channel` creates an `nt::*Publisher` (stored in a `std::vector` on the sender to keep it alive) and returns a lambda capturing it by value.

---

## `src/localization/sim_sender.h` (new)

```cpp
#pragma once
#include "localization/position_sender.h"

namespace localization {

class SimSender : public IPositionSender {
 public:
  void Send(const position_estimate_t& estimate) override;
  // Stores the last estimate for test assertions
  std::optional<position_estimate_t> last_estimate;
};

}  // namespace localization
```

Used in tests and simulation runs. `Send()` stores the estimate and optionally logs it.

---

## `src/localization/position_solver.h` — modified

Replace the existing `IJointPositionSolverNode` with a new multi-camera accumulator interface:

```cpp
class IAccumulatingSolverNode {
 public:
  // Called once per camera per iteration.
  // camera_idx: which camera's detections these are.
  // ctx: iteration context; solver holds a copy until all cameras report.
  virtual void Accumulate(
      int camera_idx,
      std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
      std::shared_ptr<utils::Context> ctx) = 0;

  virtual ~IAccumulatingSolverNode() = default;
};
```

`IPositionSolverNode` (single-camera, ambiguous) remains unchanged — it is still used internally by `UnambiguousSolverNode`.

---

## `src/localization/unambiguous_solver_node.h` — modified

```cpp
class UnambiguousSolverNode : public IAccumulatingSolverNode {
 public:
  UnambiguousSolverNode(
      const std::vector<camera::camera_constant_t>& camera_constants,
      std::unique_ptr<IPositionSender> sender,
      const wpi::apriltag::AprilTagFieldLayout& layout = kapriltag_layout);

  // IAccumulatingSolverNode
  void Accumulate(int camera_idx,
                  std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
                  std::shared_ptr<utils::Context> ctx) override;

  // Kept for unit testing without the callback machinery
  auto SolveWithoutNotify(
      const std::vector<std::vector<apriltag::tag_detection_t>>& detection_batches,
      bool reject_far_tags = true) -> std::optional<position_estimate_t>;

 private:
  // Solves with whatever is in accumulated_detections_, sends the result, and
  // resets all per-iteration state. Must be called with mutex_ held via `lock`.
  // Releases the lock internally before joining the timer thread or releasing
  // contexts to avoid deadlock.
  void SolveAndReset(std::unique_lock<std::mutex>& lock);

  // ... existing solve helpers unchanged ...

  int num_cameras_;
  std::unique_ptr<IPositionSender> sender_;

  // Per-iteration accumulation state (all under mutex_)
  std::mutex mutex_;
  int cameras_reported_{0};       // how many distinct cameras have called Accumulate
  int total_detections_{0};       // sum of detections->size() across all Accumulate calls
  bool iteration_solved_{false};  // prevents double-solve if timer and last camera race
  std::vector<std::vector<apriltag::tag_detection_t>> accumulated_detections_;
  std::vector<std::shared_ptr<utils::Context>> held_contexts_;

  // Launched once total_detections_ >= 2. Fires after kaccumulation_timeout,
  // then calls SolveAndReset with whatever cameras have reported.
  std::optional<std::jthread> timer_thread_;

  static constexpr std::chrono::milliseconds kaccumulation_timeout{/* TBD at tuning time */};

  std::optional<position_estimate_t> prev_pose_estimate_;
  // ... existing members ...
};
```

### `Accumulate()` implementation

```
Accumulate(camera_idx, detections, ctx):
  unique_lock lock(mutex_)

  accumulated_detections_[camera_idx] = *detections
  held_contexts_.push_back(ctx)
  cameras_reported_++
  total_detections_ += detections->size()

  // Start timeout once we have enough data to attempt a solve
  if total_detections_ >= 2 && !timer_thread_.has_value() && !iteration_solved_:
    timer_thread_.emplace([this](std::stop_token st) {
      std::this_thread::sleep_for(kaccumulation_timeout)
      if st.stop_requested(): return
      std::unique_lock lock(mutex_)
      if !iteration_solved_:
        SolveAndReset(lock)   // solves with however many cameras have reported
    })

  // Solve immediately once all cameras have reported, regardless of tag count
  if cameras_reported_ == num_cameras_ && !iteration_solved_:
    if timer_thread_.has_value():
      timer_thread_->request_stop()   // non-blocking; timer exits on its own
    SolveAndReset(lock)
```

### `SolveAndReset()` implementation

Called under `mutex_` via a `unique_lock`. Releases the lock before any blocking operations.

```
SolveAndReset(unique_lock& lock):
  iteration_solved_ = true

  // Move state out so destructors run after unlock, avoiding two deadlock scenarios:
  //   1. jthread destructor joins timer thread, which is blocked waiting for mutex_
  //   2. Context::~Context calls WakeUp (fine, different mutex, but cleaner outside lock)
  auto old_timer    = std::move(timer_thread_)
  auto old_contexts = std::move(held_contexts_)
  timer_thread_     = std::nullopt

  auto result = SolveWithoutNotify(accumulated_detections_)

  // Reset counters for the next iteration
  cameras_reported_ = 0
  total_detections_ = 0
  iteration_solved_ = false
  for each d in accumulated_detections_: d.clear()

  lock.unlock()

  old_timer.reset()       // request_stop + join; timer sees iteration_solved_==true and exits
  old_contexts.clear()    // releases ctx refs; last destructor calls WakeUp

  if result.has_value():
    sender_->Send(*result)
```

**Thread safety notes:**
- `mutex_` serializes all `Accumulate` calls and the timer callback; `iteration_solved_` is the once-flag so only one path runs `SolveAndReset`.
- The `jthread` destructor (which calls `request_stop` + `join`) is never invoked while `mutex_` is held. The timer thread, when it finally acquires `mutex_`, will find `iteration_solved_ == true` and return without calling `SolveAndReset` again.
- `held_contexts_` is moved out before `unlock()` so `Context::~Context` and `WakeUp()` are always called outside `mutex_`.
- The `iteration_solved_ = false` reset at the end of `SolveAndReset` happens before `unlock()`, so the next iteration starts clean.

---

## Callback Registration Pattern (wiring in `main.cc`)

```cpp
auto controller = std::make_shared<utils::LoopController>(num_cameras);

// Camera 0 setup
auto camera0 = std::make_unique<camera::UVCCameraNode>(config0);
camera0->RegisterCallback([&](std::shared_ptr<camera::JpegBuffer> frame) {
  controller->ReceiveFrame(0, frame);
});
camera0->Start();

// Camera 1 setup (same pattern)

// Streamer for camera 0 (registered as iteration callback; timestamp unused here)
auto streamer0 = std::make_unique<streamer::JpegBufferStreamerNode>("/stream0", 5800);
controller->RegisterIterationCallback(
    0,
    [&](std::shared_ptr<camera::JpegBuffer> jpeg, double, std::shared_ptr<utils::Context>) {
      streamer0->Stream(jpeg);
    });

// Decode → Detect → Solve pipeline for camera 0
auto decoder0 = std::make_unique<camera::NvjpegDecodeNode>("cam0");
auto detector0 = std::make_unique<apriltag::GPUApriltagDetectorNode>(
    width, height, intrinsics0);

detector0->RegisterCallback(
    [&solver, idx=0](auto detections, auto ctx) {
      solver->Accumulate(idx, detections, ctx);
    });

decoder0->RegisterCallback(
    [&detector0](camera::TimestampedDecodedFrame frame, auto ctx) {
      detector0->Detect(*frame.buffer, frame.timestamp, ctx);
    });

controller->RegisterIterationCallback(
    0,
    [&decoder0](auto jpeg, unsigned long timestamp, auto ctx) {
      decoder0->Decode(jpeg, timestamp, ctx);
    });

// Same for camera 1 with idx=1

// Sender
auto sender = std::make_unique<localization::NetworkTableSender>("front");

// Solver owns sender
auto solver = std::make_unique<localization::UnambiguousSolverNode>(
    camera_constants, std::move(sender));

// Start loop
stop::RegisterHandler([&]{ controller->RequestStop(); });
controller->Run();
```

---

## Timestamp Handling

The jpeg capture timestamp (`uvc_frame_t->capture_time`, converted to microseconds as an `unsigned long`) is the single authoritative timestamp for a frame. It propagates as follows:

```cpp
UVCCameraNode::CallBack()
  → LoopController::ReceiveFrame(..., timestamp)       stored in latest_timestamps_[i]
  → iteration callback(jpeg, timestamp, ctx)
  → NvjpegDecodeNode::Decode(jpeg, timestamp, ctx)
  → NvjpegDecodeNode::DecodeJpegBuffer(...)
  → callbacks_(TimestampedDecodedFrame{buffer, timestamp}, ctx)
  → IApriltagDetectorNode::Detect(frame, timestamp, ctx)
  → tag_detection_t.timestamp = timestamp              (set per detection)
  → UnambiguousSolverNode::SolveWithoutNotify()
      GetAmbiguousEstimates: filters detections by timestamp recency
      SolveWithoutNotify: averages timestamps → position_estimate_t.timestamp
  → IPositionSender::Send(estimate)                   estimate.timestamp is the capture time
```

The timestamp is never re-derived or replaced at intermediate steps. Decode time and solve time are not recorded. The `UnambiguousSolverNode` existing staleness check (`latest_timestamp - detections[0].timestamp >= kacceptable_frame_recency`) works correctly because all `tag_detection_t` objects from a given camera share the same capture timestamp.

---

## Handling Cameras With No New Frame

**Decision for plan: always call all N camera callbacks. Decoder returns immediately and calls detector with empty detections if jpeg is null. Solver always expects exactly N calls.**

---

## Debug / Development Hooks

Per the design doc, no production debug info is baked in. However, the design supports easy hookup during development:

- Any node that receives a `shared_ptr<Context>` can also register additional callbacks. E.g., register a second callback on the decoder to receive decoded frames for display.
- The `JpegBufferStreamerNode` is already a direct hook: register it as an iteration callback on LoopController without threading it through decode/detect.
- To log raw detections during development: add a second callback on the detector that receives `(detections, ctx)` and writes to a file.
- The callback registration pattern in `main.cc` is the extension point — no changes to node internals required.

---

## `src/localization/multi_tag_solver_node.h/.cc` — unchanged

Used internally by `UnambiguousSolverNode::SolveWithoutNotify`. Its `AmbiguousSolveWithoutNotify` method is a pure function (no callbacks fired internally). No changes needed.

---

## `src/localization/square_solver_node.h/.cc` — unchanged

Used by `MultiTagSolverNode`. No changes.

---

## `src/streamer/jpeg_buffer_streamer_node.h/.cc` — unchanged

`Stream()` is synchronous and doesn't need context. Called from `LoopController` iteration callback as a fire-and-forget sink. Context ref is captured by the lambda but the streamer itself doesn't need it.

---

## `src/utils/stop.h/.cc` — modified slightly

`RegisterHandler` should accept a callable so `main.cc` can pass `[&]{ controller->RequestStop(); }`. If the current implementation only supports SIGINT/SIGTERM with a global flag, update it to accept an additional hook function:

```cpp
void RegisterHandler(std::function<void()> on_stop = {});
```

If the current API is already flexible enough, no change needed.

---

## CMakeLists changes

- `src/utils/CMakeLists.txt`: add `loop_controller.cc`.
- `src/localization/CMakeLists.txt`: add `networktable_sender.cc`; remove bos-specific deps if any.
- `src/main/CMakeLists.txt`: add `main.cc`, link all node libraries.
- New `IPositionSender`, `SimSender` are header-only or added to existing targets.

---

## Testing Strategy

### Unit tests (per-node, no camera hardware)

Each node can be tested in isolation because:
- Inputs are explicit method calls (`Decode()`, `Detect()`, `Accumulate()`).
- Outputs are registered callbacks, which tests can register with assertions.
- `Context` can be constructed directly in tests; its destructor is harmless if no LoopController is attached.

Example: test `UnambiguousSolverNode` by constructing it with a `SimSender`, calling `Accumulate()` N times with mock detections, and asserting `SimSender::last_estimate`.

### Integration test (no camera hardware)

Replace `UVCCameraNode` with a `DiskCameraNode` (like `bos/src/camera/disk_camera`) that reads jpegs from disk and calls `LoopController::ReceiveFrame()`. Wire the full pipeline. Assert final `position_estimate_t` values.

### Swappable nodes

- `GPUApriltagDetectorNode` ↔ `OpenCVApriltagDetectorNode`: both implement `IApriltagDetectorNode` (`INode<shared_ptr<vector<tag_detection_t>>>`). Swap by changing `main.cc`.
- `NetworkTableSender` ↔ `SimSender`: both implement `IPositionSender`. Swap by changing what's passed to `UnambiguousSolverNode`.
- Any decode node (nvjpeg ↔ opencv decode): if an `OpenCVDecodeNode` is added later, it implements `INode<shared_ptr<DecodedJpegNvBuffer>>` (or a cv::Mat equivalent) with `Decode(jpeg, ctx)` and the same callback contract.
