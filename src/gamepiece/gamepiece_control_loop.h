#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include "control_loop/context.h"
#include "control_loop/node.h"

namespace camera {
class DecodedJpegBuffer;
}

namespace gamepiece {

// Runs gamepiece nodes independently from localization while retaining only
// the latest decoded frame from each registered decoder channel.
// Scheduling and waits run on a dedicated thread; asynchronous nodes submit
// their work to the same ThreadPool supplied to the localization nodes.
class GamepieceControlLoop {
 public:
  explicit GamepieceControlLoop(
      std::chrono::milliseconds period = std::chrono::milliseconds(20));
  ~GamepieceControlLoop();

  GamepieceControlLoop(const GamepieceControlLoop&) = delete;
  auto operator=(const GamepieceControlLoop&) -> GamepieceControlLoop& = delete;

  void RegisterDecodedFrameSource(
      const std::shared_ptr<control_loop::INode>& decoder,
      std::string_view decoded_channel);
  void RegisterNode(const std::shared_ptr<control_loop::INode>& node);

  void Start();
  void Stop();

 private:
  struct DecodedFrameState;

  void ValidateNodeGraph();
  void RegisterNodeCallbacks();
  void Run(std::stop_token stop_token);

  std::shared_ptr<DecodedFrameState> decoded_frame_state_;
  std::chrono::milliseconds period_;
  std::vector<std::string> decoded_channels_in_order_;
  std::vector<std::vector<
      std::function<void(const control_loop::Context&)>>>
      decoded_frame_callbacks_;
  std::vector<std::shared_ptr<control_loop::INode>> nodes_;
  std::unordered_set<std::string> decoded_channels_;
  std::jthread thread_;
  bool started_ = false;
};

}  // namespace gamepiece
