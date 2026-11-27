#include "streamer/networktables_command_streamer_node.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "absl/log/log.h"

namespace streamer {
using pathing::CommandMessage;
using CommandStatus = CommandMessage::Status;

namespace {

auto StatusName(CommandStatus status) -> const char* {
  switch (status) {
    case CommandStatus::Inactive:
      return "Inactive";
    case CommandStatus::Following:
      return "Following";
    case CommandStatus::ReachedGoal:
      return "ReachedGoal";
    case CommandStatus::InvalidInput:
      return "InvalidInput";
    case CommandStatus::StaleInput:
      return "StaleInput";
    case CommandStatus::NoPath:
      return "NoPath";
    case CommandStatus::FitFailed:
      return "FitFailed";
    case CommandStatus::Disconnected:
      return "Disconnected";
  }
  return "InvalidInput";
}

}  // namespace

NetworkTablesCommandStreamerNode::NetworkTablesCommandStreamerNode(
    std::string_view input_channel, std::string_view command_topic,
    std::chrono::milliseconds period, nt::NetworkTableInstance instance)
    : input_channel_(input_channel),
      dependencies_{{input_channel, typeid(CommandMessage)}},
      instance_(instance),
      publisher_(instance.GetDoubleArrayTopic(command_topic).Publish(
          {.periodic = std::chrono::duration<double>(period).count(),
           .keepDuplicates = true})) {}

auto NetworkTablesCommandStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    if (!context->Exists(input_channel_))
      return;
    const auto* command = context->GetMessage<CommandMessage>(input_channel_);
    const CommandMessage empty_command;
    Publish(command ? *command : empty_command);
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

auto NetworkTablesCommandStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto NetworkTablesCommandStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  static const std::vector<control_loop::MessageDescriptor> publications;
  return publications;
}

void NetworkTablesCommandStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

void NetworkTablesCommandStreamerNode::PublishStop() {
  Publish(CommandMessage{});
}

void NetworkTablesCommandStreamerNode::Publish(const CommandMessage& command) {
  std::scoped_lock lock(mutex_);
  auto status = command.status;
  if (!command.input_timestamps && (status == CommandStatus::Following ||
                                    status == CommandStatus::ReachedGoal)) {
    status = CommandStatus::InvalidInput;
  }
  if (!std::isfinite(command.vx) || !std::isfinite(command.vy) ||
      !std::isfinite(command.omega))
    status = CommandStatus::InvalidInput;
  const auto offset = instance_.GetServerTimeOffset();
  if (!instance_.IsConnected() || !offset)
    status = CommandStatus::Disconnected;
  const std::int64_t now = nt::Now();
  const bool following = status == CommandStatus::Following;
  const auto timestamps =
      command.input_timestamps.value_or(CommandMessage::InputTimestamps{});
  const std::array<double, 7> frame{
      following ? command.vx : 0,
      following ? command.vy : 0,
      following ? command.omega : 0,
      static_cast<double>(status),
      static_cast<double>(std::max(std::int64_t{0}, timestamps.pose_server_us)),
      static_cast<double>(std::max(std::int64_t{0}, timestamps.goal_server_us)),
      offset ? static_cast<double>(now + *offset) : 0};
  publisher_.Set(frame, now);
  instance_.Flush();
  if (last_status_ != status) {
    LOG(INFO) << "Pathing: " << StatusName(status);
    last_status_ = status;
  }
}

}  // namespace streamer
