#include "streamer/networktables_command_streamer_node.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include "absl/log/log.h"
#include "localization/robot_pose_node.h"

namespace streamer {
using pathing::CommandMessage;
using pathing::PathingStatus;

namespace {

auto StatusName(PathingStatus status) -> const char* {
  switch (status) {
    case PathingStatus::Inactive:
      return "Inactive";
    case PathingStatus::Following:
      return "Following";
    case PathingStatus::ReachedGoal:
      return "ReachedGoal";
    case PathingStatus::InvalidInput:
      return "InvalidInput";
    case PathingStatus::StaleInput:
      return "StaleInput";
    case PathingStatus::NoPath:
      return "NoPath";
    case PathingStatus::FitFailed:
      return "FitFailed";
    case PathingStatus::Disconnected:
      return "Disconnected";
  }
  return "InvalidInput";
}

}  // namespace

NetworkTablesCommandStreamerNode::NetworkTablesCommandStreamerNode(
    std::string_view input_channel, std::string_view command_topic,
    std::chrono::milliseconds period, nt::NetworkTableInstance instance,
    std::shared_ptr<const localization::RobotPoseNode> pose_node)
    : input_channel_(input_channel),
      dependencies_{{input_channel, typeid(CommandMessage)}},
      instance_(instance),
      pose_node_(std::move(pose_node)),
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
  if (command.input) {
    const auto validation = pose_node_->ValidateInput(*command.input);
    if (validation != PathingStatus::Following)
      status = validation;
  } else if (status == PathingStatus::Following ||
             status == PathingStatus::ReachedGoal) {
    status = PathingStatus::InvalidInput;
  }
  if (!std::isfinite(command.vx) || !std::isfinite(command.vy) ||
      !std::isfinite(command.omega))
    status = PathingStatus::InvalidInput;
  const auto offset = instance_.GetServerTimeOffset();
  if (!instance_.IsConnected() || !offset)
    status = PathingStatus::Disconnected;
  const std::int64_t now = nt::Now();
  const bool following = status == PathingStatus::Following;
  const std::array<double, 7> frame{
      following ? command.vx : 0,
      following ? command.vy : 0,
      following ? command.omega : 0,
      static_cast<double>(status),
      command.input
          ? static_cast<double>(std::max(std::int64_t{0},
                                         command.input->pose_server_us))
          : 0,
      command.input
          ? static_cast<double>(std::max(std::int64_t{0},
                                         command.input->goal_server_us))
          : 0,
      offset ? static_cast<double>(now + *offset) : 0};
  publisher_.Set(frame, now);
  instance_.Flush();
  if (last_status_ != status) {
    LOG(INFO) << "Pathing: " << StatusName(status);
    last_status_ = status;
  }
}

}  // namespace streamer
