#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <networktables/DoubleArrayTopic.h>
#include <networktables/NetworkTableInstance.h>

#include "control_loop/node.h"
#include "pathing/command_message.h"

namespace streamer {

// Validates a pathing command and publishes it to the robot over NetworkTables.
class NetworkTablesCommandStreamerNode final : public control_loop::INode {
 public:
  NetworkTablesCommandStreamerNode(
      std::string_view input_channel, std::string_view command_topic,
      std::chrono::milliseconds period, nt::NetworkTableInstance instance);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) override;

  void PublishStop();

 private:
  void Publish(const pathing::CommandMessage& command);

  std::string input_channel_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  nt::NetworkTableInstance instance_;
  nt::DoubleArrayPublisher publisher_;
  std::mutex mutex_;
  std::optional<pathing::PathingStatus> last_status_;
};

}  // namespace streamer
