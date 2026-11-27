#include "localization/robot_pose_node.h"

#include <cmath>
#include <memory>
#include <utility>

#include "pathing/pose_message.h"

namespace localization {
using pathing::PathingStatus;
using pathing::PoseMessage;

namespace {
auto FinitePose(const frc::Pose2d& pose) -> bool {
  return std::isfinite(pose.X().value()) && std::isfinite(pose.Y().value()) &&
         std::isfinite(pose.Rotation().Radians().value());
}
}  // namespace

RobotPoseNode::RobotPoseNode(nt::NetworkTableInstance instance,
                             std::string_view pose_topic,
                             std::string_view goal_topic,
                             std::chrono::milliseconds period)
    : publications_{{kPose, typeid(PoseMessage)},
                    {kGoal, typeid(PoseMessage)},
                    {kInput, typeid(PathingInputMessage)}},
      instance_(instance),
      pose_(instance.GetStructTopic<frc::Pose2d>(pose_topic)
                .Subscribe(
                    frc::Pose2d{},
                    {.periodic = std::chrono::duration<double>(period).count(),
                     .keepDuplicates = true})),
      goal_(instance.GetStructTopic<frc::Pose2d>(goal_topic)
                .Subscribe(
                    frc::Pose2d{},
                    {.periodic = std::chrono::duration<double>(period).count(),
                     .keepDuplicates = true})) {}

auto RobotPoseNode::ValidateInput(const PathingInputMessage& input) const
    -> PathingStatus {
  if (!instance_.IsConnected() || !instance_.GetServerTimeOffset())
    return PathingStatus::Disconnected;
  if (!pose_.GetTopic().Exists() || !goal_.GetTopic().Exists())
    return PathingStatus::Inactive;
  if (pose_.GetTopic().GetTypeString() != "struct:Pose2d" ||
      goal_.GetTopic().GetTypeString() != "struct:Pose2d")
    return PathingStatus::InvalidInput;
  if (input.pose_local_us == 0 || input.goal_local_us == 0)
    return PathingStatus::Inactive;
  if (input.pose_server_us <= 0 || input.goal_server_us <= 0)
    return PathingStatus::InvalidInput;
  if (!FinitePose(input.pose) || !FinitePose(input.goal))
    return PathingStatus::InvalidInput;
  return PathingStatus::Following;
}

auto RobotPoseNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    {
      std::scoped_lock lock(mutex_);
      if (context->Exists(kInput))
        return;
      const auto pose = pose_.GetAtomic();
      const auto goal = goal_.GetAtomic();
      auto input = std::make_shared<PathingInputMessage>();
      input->pose = pose.value;
      input->goal = goal.value;
      input->pose_local_us = pose.time;
      input->goal_local_us = goal.time;
      input->pose_server_us = pose.serverTime;
      input->goal_server_us = goal.serverTime;
      input->status = ValidateInput(*input);
      if (input->status == PathingStatus::Following) {
        context->SetMessage(kPose, std::make_shared<PoseMessage>(pose.value));
        context->SetMessage(kGoal, std::make_shared<PoseMessage>(goal.value));
      } else {
        context->SetMessage(kPose, nullptr);
        context->SetMessage(kGoal, nullptr);
      }
      context->SetMessage(kInput, std::move(input));
    }
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

auto RobotPoseNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  static const std::vector<control_loop::MessageDescriptor> dependencies;
  return dependencies;
}

auto RobotPoseNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void RobotPoseNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace localization
