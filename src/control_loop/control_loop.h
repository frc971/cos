#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include "control_loop/context.h"
#include "control_loop/node.h"

namespace logging {
class WPILogWriter;
}

namespace control_loop {

class ControlLoop {
 public:
  ControlLoop(
      std::chrono::milliseconds frequency = std::chrono::milliseconds(10));
  void RegisterCallback(const std::function<void(const Context&)>& callback);
  void RegisterNode(const std::shared_ptr<INode>& node);
  void RegisterDependencyNode(const std::shared_ptr<INode>& node);
  void EnableLatencyLog();
  [[nodiscard]] auto GetLogPublications() const -> std::vector<MessageDescriptor>;
  void EnableWPILog(std::shared_ptr<logging::WPILogWriter> writer);
  void Start();
  void Stop();
  [[nodiscard]] auto GetLoopsPerSecond() const -> double;
  void SetMaxContext(size_t max_contexts);

 private:
  void ValidateNodeGraph();
  void RegisterNodeCallbacks();

 private:
  std::jthread thread_;
  std::chrono::milliseconds period_;
  std::vector<std::function<void(Context)>> callbacks_;
  std::vector<std::function<void(Context)>> dependencies_;
  std::vector<std::shared_ptr<INode>> nodes_;
  std::vector<std::shared_ptr<INode>> dependency_nodes_;
  bool log_latency_ = false;
  std::queue<std::chrono::steady_clock::time_point> timestamp_queue_;
  std::atomic<double> loops_per_second_ = -1;
  std::vector<std::shared_ptr<ContextInternal>> contexts_;
  size_t max_contexts_ = 1;
  std::uint64_t loop_count_ = 0;
  std::shared_ptr<logging::WPILogWriter> wpilog_writer_;

 private:
  static const size_t kTimestampQueueMaxSize = 100;
  constexpr static const double kMinLoopSeconds = 0.001;
};

}  // namespace control_loop
