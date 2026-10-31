#include "streamer/ball_velocity_goal_streamer_node.h"

#include <array>
#include <memory>

#include <networktables/NetworkTable.h>

#include "gamepiece/ball_pathing_node.h"

namespace streamer {
namespace {
constexpr nt::PubSubOptions kPublishOptions = {
    .pollStorage = 20, .sendAll = true, .keepDuplicates = true};
}

BallVelocityGoalStreamerNode::BallVelocityGoalStreamerNode(
    std::string_view input_path, std::string_view networktable_path)
    : input_path_(input_path),
      instance_(nt::NetworkTableInstance::GetDefault()),
      dependencies_({{input_path_,
                      typeid(gamepiece::BallVelocityGoalMessage)}}) {
  const std::shared_ptr<nt::NetworkTable> table =
      instance_.GetTable(networktable_path);
  publisher_ = table->GetDoubleArrayTopic("BallVelocityGoal")
                   .Publish(kPublishOptions);
}

auto BallVelocityGoalStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto* goal =
        context->GetMessage<gamepiece::BallVelocityGoalMessage>(input_path_);
    if (goal != nullptr) {
      const std::array<double, 9> values{
          goal->vx_mps,
          goal->vy_mps,
          goal->target_x_m,
          goal->target_y_m,
          static_cast<double>(goal->estimated_ball_count),
          goal->valid ? 1.0 : 0.0,
          goal->held ? 1.0 : 0.0,
          goal->timestamp,
          goal->ball_evidence_m2,
      };
      publisher_.Set(values);
      instance_.Flush();
    }
    for (const auto& callback : callbacks_) callback(context);
  };
}

void BallVelocityGoalStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto BallVelocityGoalStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto BallVelocityGoalStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace streamer
