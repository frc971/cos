#include "localization/robot_pose_receiver_node.h"

#include <memory>
#include <utility>

#include "pathing/pose_message.h"

namespace localization {
using pathing::PathingStatus;
using pathing::PoseMessage;

RobotPoseReceiverNode::RobotPoseReceiverNode(
    nt::NetworkTableInstance instance, std::string_view pose_topic,
    std::string_view goal_topic, std::chrono::milliseconds period)
    : publications_{{kPose, typeid(PoseMessage)},
                    {kGoal, typeid(PoseMessage)},
                    {kInput, typeid(PathingInputMessage)}},
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

auto RobotPoseReceiverNode::CreateCallback()
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
      // A cached sample is unavailable after its topic is unpublished.
      if (pose.time != 0 && goal.time != 0 && pose_.GetTopic().Exists() &&
          goal_.GetTopic().Exists()) {
        input->status = PathingStatus::Following;
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

auto RobotPoseReceiverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  static const std::vector<control_loop::MessageDescriptor> dependencies;
  return dependencies;
}

auto RobotPoseReceiverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void RobotPoseReceiverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace localization
