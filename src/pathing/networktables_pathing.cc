#include "pathing/networktables_pathing.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "control_loop/control_loop.h"
#include "localization/robot_pose_node.h"
#include "pathing/command_message.h"
#include "pathing/navgrid_source_node.h"
#include "pathing/pose_message.h"
#include "streamer/navgrid_streamer_node.h"
#include "streamer/networktables_command_streamer_node.h"

namespace pathing {
using nt::NetworkTableInstance;

namespace {
using localization::PathingInputMessage;
using localization::RobotPoseNode;
constexpr char kSpline[] = "pathing:spline";
constexpr char kCommand[] = "pathing:command";
constexpr char kSplineOverlay[] = "pathing:spline_overlay";

class Node : public control_loop::INode {
 public:
  Node(std::vector<control_loop::MessageDescriptor> dependencies,
       std::vector<control_loop::MessageDescriptor> publications)
      : dependencies_(std::move(dependencies)),
        publications_(std::move(publications)) {}
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override {
    callbacks_.push_back(callback);
  }

 protected:
  void Notify(const control_loop::Context& context) const {
    for (const auto& callback : callbacks_)
      callback(context);
  }
  std::mutex mutex_;

 private:
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

class FollowerNode final : public Node {
 public:
  explicit FollowerNode(NetworkTablesPathingSettings settings)
      : Node({{RobotPoseNode::kPose, typeid(PoseMessage)},
              {RobotPoseNode::kGoal, typeid(PoseMessage)},
              {RobotPoseNode::kInput, typeid(PathingInputMessage)},
              {kSpline, typeid(SplinePathMessage)}},
             {{kCommand, typeid(CommandMessage)}}),
        settings_(std::move(settings)) {}

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) -> void {
      {
        std::scoped_lock lock(mutex_);
        if (!context->Exists(RobotPoseNode::kPose) ||
            !context->Exists(RobotPoseNode::kGoal) ||
            !context->Exists(RobotPoseNode::kInput) ||
            !context->Exists(kSpline) || context->Exists(kCommand))
          return;
        auto command = std::make_shared<CommandMessage>();
        command->input = context->GetSharedMessage<PathingInputMessage>(
            RobotPoseNode::kInput);
        if (command->input) {
          command->status = command->input->status;
          if (command->status == PathingStatus::Following)
            Calculate(context, *command);
        }
        context->SetMessage(kCommand, std::move(command));
      }
      Notify(context);
    };
  }

 private:
  void Calculate(const control_loop::Context& context,
                 CommandMessage& command) {
    const auto* pose = context->GetMessage<PoseMessage>(RobotPoseNode::kPose);
    const auto* goal = context->GetMessage<PoseMessage>(RobotPoseNode::kGoal);
    const auto* spline = context->GetMessage<SplinePathMessage>(kSpline);
    command.status = PathingStatus::InvalidInput;
    if (!pose || !goal || !spline || !spline->planner)
      return;
    const auto& planner = *spline->planner;
    const auto cleared = planner.BlockedStartCells(pose->goal_position);
    if (!planner.IsValidStartPose(pose->goal_position, cleared) ||
        !planner.IsValidPose(goal->goal_position, cleared))
      return;
    switch (spline->status) {
      case SplineStatus::InvalidInput:
        return;
      case SplineStatus::NoPath:
        command.status = PathingStatus::NoPath;
        return;
      case SplineStatus::FitFailed:
        command.status = PathingStatus::FitFailed;
        return;
      case SplineStatus::Ready:
        break;
    }
    const auto& current = pose->goal_position;
    const auto& target = goal->goal_position;
    const auto& motion = settings_.motion;
    const double heading_error =
        std::remainder(target.Rotation().Radians().value() -
                           current.Rotation().Radians().value(),
                       2 * std::numbers::pi);
    const bool at_position =
        current.Translation().Distance(target.Translation()) <=
        motion.position_tolerance;
    const bool at_heading =
        motion.max_rotation.value() == 0 ||
        std::abs(heading_error) <= motion.heading_tolerance.value();
    command.status = at_position && at_heading ? PathingStatus::ReachedGoal
                                               : PathingStatus::Following;
    if (command.status == PathingStatus::ReachedGoal)
      return;
    if (!at_position) {
      const auto& samples = spline->samples;
      if (samples.size() < 2 ||
          samples.front().Distance(current.Translation()).value() > 1e-6) {
        command.status = PathingStatus::FitFailed;
        return;
      }
      double remaining = 0;
      for (size_t i = 1; i < samples.size(); ++i)
        remaining += samples[i].Distance(samples[i - 1]).value();
      double walked = 0;
      std::optional<frc::Translation2d> lookahead;
      for (size_t i = 1; i < samples.size(); ++i) {
        const double length = samples[i].Distance(samples[i - 1]).value();
        if (length == 0)
          continue;
        const double distance =
            std::min(length, settings_.lookahead.value() - walked);
        const auto point = samples[i - 1] +
                           (samples[i] - samples[i - 1]) * (distance / length);
        if (planner.IsVisible(current.X().value(), current.Y().value(),
                              point.X().value(), point.Y().value(), cleared))
          lookahead = point;
        walked += length;
        if (walked >= settings_.lookahead.value())
          break;
      }
      const double distance =
          lookahead ? lookahead->Distance(current.Translation()).value() : 0;
      if (distance <= 0) {
        command.status = PathingStatus::FitFailed;
        return;
      }
      const double hold_seconds =
          std::chrono::duration<double>(settings_.command_timeout).count();
      const double speed = std::min({motion.max_translation.value(),
                                     motion.translation_gain * remaining,
                                     distance / hold_seconds});
      command.vx = (lookahead->X() - current.X()).value() * speed / distance;
      command.vy = (lookahead->Y() - current.Y()).value() * speed / distance;
    }
    if (!at_heading)
      command.omega =
          std::clamp(motion.heading_gain * heading_error,
                     -motion.max_rotation.value(), motion.max_rotation.value());
  }

  NetworkTablesPathingSettings settings_;
};

auto Positive(double value) -> bool {
  return std::isfinite(value) && value > 0;
}

}  // namespace

void NetworkTablesPathingSettings::Validate() const {
  if (navgrid_streamer &&
      (navgrid_streamer_port < 1 || navgrid_streamer_port > 65535))
    throw std::invalid_argument(
        "Navgrid streamer port must be between 1 and 65535");
  if (period.count() <= 0 || command_timeout.count() <= 0 ||
      period > command_timeout || !Positive(lookahead.value()) ||
      !Positive(motion.max_translation.value()) ||
      !std::isfinite(motion.max_rotation.value()) ||
      motion.max_rotation.value() < 0 || !Positive(motion.translation_gain) ||
      !Positive(motion.heading_gain) ||
      !Positive(motion.position_tolerance.value()) ||
      !Positive(motion.heading_tolerance.value()))
    throw std::invalid_argument("Invalid NetworkTables pathing settings");
  if (pose_topic.empty() || goal_topic.empty() || command_topic.empty() ||
      pose_topic.front() != '/' || goal_topic.front() != '/' ||
      command_topic.front() != '/' || pose_topic == goal_topic ||
      command_topic == pose_topic || command_topic == goal_topic)
    throw std::invalid_argument(
        "Pathing NT topics must be absolute and distinct");
}

class NetworkTablesPathing::Impl {
 public:
  Impl(const streamer::NavgridMessage& navgrid,
       const NetworkTablesPathingSettings& settings,
       nt::NetworkTableInstance instance)
      : loop(settings.period) {
    settings.Validate();
    auto input = std::make_shared<RobotPoseNode>(
        instance, settings.pose_topic, settings.goal_topic, settings.period);
    auto spline_settings = settings.spline;
    spline_settings.robot_radius_meters = settings.robot_radius_meters;
    auto spline = std::make_shared<DijkstraSplineNode>(
        RobotPoseNode::kPose, RobotPoseNode::kGoal, kSpline,
        NavgridSourceNode::kNavgrid, spline_settings);
    if (settings.navgrid_streamer)
      spline->EnableSplineDebug(kSplineOverlay);
    auto follower = std::make_shared<FollowerNode>(settings);
    output = std::make_shared<streamer::NetworkTablesCommandStreamerNode>(
        kCommand, settings.command_topic, settings.period, instance, input);
    loop.SetMaxContext(1);
    loop.RegisterDependencyNode(std::make_shared<NavgridSourceNode>(navgrid));
    loop.RegisterDependencyNode(input);
    loop.RegisterNode(spline);
    loop.RegisterNode(follower);
    loop.RegisterNode(output);
    if (settings.navgrid_streamer) {
      auto streamer_node = std::make_shared<streamer::NavgridStreamerNode>(
          "/navgrid", settings.navgrid_streamer_port,
          settings.field_image_path);
      streamer_node->AddPathingInputs(kSpline, RobotPoseNode::kInput,
                                      RobotPoseNode::kPose, kCommand);
      streamer_node->AddPolyline(kSplineOverlay);
      loop.RegisterNode(streamer_node);
    }
  }
  control_loop::ControlLoop loop;
  std::shared_ptr<streamer::NetworkTablesCommandStreamerNode> output;
  bool started = false;
};

NetworkTablesPathing::NetworkTablesPathing(
    const streamer::NavgridMessage& navgrid,
    NetworkTablesPathingSettings settings, NetworkTableInstance instance)
    : impl_(std::make_unique<Impl>(navgrid, settings, instance)) {}

NetworkTablesPathing::~NetworkTablesPathing() {
  Stop();
}

void NetworkTablesPathing::Start() {
  if (impl_->started)
    return;
  impl_->loop.Start();
  impl_->started = true;
}

void NetworkTablesPathing::Stop() {
  if (!impl_->started)
    return;
  impl_->loop.Stop();
  impl_->started = false;
  impl_->output->PublishStop();
}
}  // namespace pathing
