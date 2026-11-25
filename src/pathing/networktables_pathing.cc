#include "pathing/networktables_pathing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <frc/geometry/struct/Pose2dStruct.h>
#include <networktables/DoubleArrayTopic.h>
#include <networktables/StructTopic.h>

#include "absl/log/log.h"
#include "control_loop/control_loop.h"
#include "streamer/navgrid_streamer_node.h"

namespace pathing {
namespace {
constexpr char kPose[] = "pathing:pose";
constexpr char kGoal[] = "pathing:goal";
constexpr char kInput[] = "pathing:input";
constexpr char kSpline[] = "pathing:spline";
constexpr char kCommand[] = "pathing:command";
constexpr char kGrid[] = "pathing:navgrid";
constexpr char kRouteOverlay[] = "pathing:route_overlay";
constexpr char kSplineOverlay[] = "pathing:spline_overlay";
constexpr char kVelocityOverlay[] = "pathing:velocity_overlay";
using Clock = NetworkTablesPathing::Clock;

class Node : public control_loop::INode {
 public:
  Node(std::vector<control_loop::MessageDescriptor> dependencies,
       std::vector<control_loop::MessageDescriptor> publications)
      : dependencies_(std::move(dependencies)),
        publications_(std::move(publications)) {}
  auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }
  auto GetPublications() const
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

class InputMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(InputMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  PathingStatus status = PathingStatus::Inactive;
  frc::Pose2d pose;
  frc::Pose2d goal;
  int64_t pose_local_us = 0;
  int64_t goal_local_us = 0;
  int64_t pose_server_us = 0;
  int64_t goal_server_us = 0;
};

class CommandMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(CommandMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  double vx = 0;
  double vy = 0;
  double omega = 0;
  PathingStatus status = PathingStatus::Inactive;
  std::shared_ptr<const InputMessage> input;
};

auto FinitePose(const frc::Pose2d& pose) -> bool {
  return std::isfinite(pose.X().value()) && std::isfinite(pose.Y().value()) &&
         std::isfinite(pose.Rotation().Radians().value());
}

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

class InputNode final : public Node {
 public:
  InputNode(const NetworkTablesPathingSettings& settings,
            nt::NetworkTableInstance instance)
      : Node({}, {{kPose, typeid(PoseMessage)},
                  {kGoal, typeid(PoseMessage)},
                  {kInput, typeid(InputMessage)}}),
        instance_(instance),
        pose_(
            instance.GetStructTopic<frc::Pose2d>(settings.pose_topic)
                .Subscribe(
                    frc::Pose2d{},
                    {.periodic =
                         std::chrono::duration<double>(settings.period).count(),
                     .keepDuplicates = true})),
        goal_(
            instance.GetStructTopic<frc::Pose2d>(settings.goal_topic)
                .Subscribe(
                    frc::Pose2d{},
                    {.periodic =
                         std::chrono::duration<double>(settings.period).count(),
                     .keepDuplicates = true})) {}

  auto ValidateInput(const InputMessage& input) const -> PathingStatus {
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

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      {
        std::scoped_lock lock(mutex_);
        if (context->Exists(kInput))
          return;
        const auto pose = pose_.GetAtomic();
        const auto goal = goal_.GetAtomic();
        auto input = std::make_shared<InputMessage>();
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
      Notify(context);
    };
  }

 private:
  nt::NetworkTableInstance instance_;
  nt::StructSubscriber<frc::Pose2d> pose_;
  nt::StructSubscriber<frc::Pose2d> goal_;
};

class FollowerNode final : public Node {
 public:
  explicit FollowerNode(const NetworkTablesPathingSettings& settings)
      : Node({{kPose, typeid(PoseMessage)},
              {kGoal, typeid(PoseMessage)},
              {kInput, typeid(InputMessage)},
              {kSpline, typeid(SplinePathMessage)}},
             {{kCommand, typeid(CommandMessage)}}),
        planner_(settings.navgrid_path, settings.spline.turn_penalty,
                 settings.robot_radius_meters),
        settings_(settings) {}

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      {
        std::scoped_lock lock(mutex_);
        if (!context->Exists(kPose) || !context->Exists(kGoal) ||
            !context->Exists(kInput) || !context->Exists(kSpline) ||
            context->Exists(kCommand))
          return;
        auto command = std::make_shared<CommandMessage>();
        command->input = context->GetSharedMessage<InputMessage>(kInput);
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
    const auto* pose = context->GetMessage<PoseMessage>(kPose);
    const auto* goal = context->GetMessage<PoseMessage>(kGoal);
    const auto* spline = context->GetMessage<SplinePathMessage>(kSpline);
    command.status = PathingStatus::InvalidInput;
    if (!pose || !goal || !spline)
      return;
    const auto cleared = planner_.BlockedStartCells(pose->goal_position);
    if (!planner_.IsValidStartPose(pose->goal_position, cleared) ||
        !planner_.IsValidPose(goal->goal_position, cleared))
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
        if (planner_.IsVisible(current.X().value(), current.Y().value(),
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

  DijkstraPlanner planner_;
  NetworkTablesPathingSettings settings_;
};

class OutputNode final : public Node {
 public:
  OutputNode(const NetworkTablesPathingSettings& settings,
             nt::NetworkTableInstance instance, Clock clock,
             std::shared_ptr<InputNode> input)
      : Node({{kCommand, typeid(CommandMessage)}}, {}),
        instance_(instance),
        clock_(std::move(clock)),
        input_(std::move(input)),
        publisher_(
            instance.GetDoubleArrayTopic(settings.command_topic)
                .Publish(
                    {.periodic =
                         std::chrono::duration<double>(settings.period).count(),
                     .keepDuplicates = true})) {}

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      if (!context->Exists(kCommand))
        return;
      const auto* command = context->GetMessage<CommandMessage>(kCommand);
      if (command)
        Publish(*command);
      else
        Publish(CommandMessage{});
      Notify(context);
    };
  }

  void PublishStop() { Publish(CommandMessage{}); }

 private:
  void Publish(const CommandMessage& command) {
    std::scoped_lock lock(mutex_);
    auto status = command.status;
    if (command.input) {
      const auto validation = input_->ValidateInput(*command.input);
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
    const int64_t now = clock_();
    const bool following = status == PathingStatus::Following;
    const std::array<double, 7> frame{
        following ? command.vx : 0,
        following ? command.vy : 0,
        following ? command.omega : 0,
        static_cast<double>(status),
        command.input ? static_cast<double>(
                            std::max(int64_t{0}, command.input->pose_server_us))
                      : 0,
        command.input ? static_cast<double>(
                            std::max(int64_t{0}, command.input->goal_server_us))
                      : 0,
        offset ? static_cast<double>(now + *offset) : 0};
    publisher_.Set(frame, now);
    instance_.Flush();
    if (last_status_ != status) {
      LOG(INFO) << "Pathing: " << StatusName(status);
      last_status_ = status;
    }
  }

  nt::NetworkTableInstance instance_;
  Clock clock_;
  std::shared_ptr<InputNode> input_;
  nt::DoubleArrayPublisher publisher_;
  std::optional<PathingStatus> last_status_;
};

// Own the streamer rather than registering it as a graph node: all layers must
// be assembled before drawing, and one frame should encode once, not once per
// dependency notification. NT command publication is registered before this node.
class VisualizationNode final : public Node {
 public:
  VisualizationNode(const NetworkTablesPathingSettings& settings,
                    std::shared_ptr<InputNode> input)
      : Node({{kPose, typeid(PoseMessage)},
              {kInput, typeid(InputMessage)},
              {kSpline, typeid(SplinePathMessage)},
              {kSplineOverlay, typeid(streamer::PolylineMessage)},
              {kCommand, typeid(CommandMessage)}},
             {{kGrid, typeid(streamer::NavgridMessage)},
              {kRouteOverlay, typeid(streamer::NavgridMessage)},
              {kVelocityOverlay, typeid(streamer::PolylineMessage)}}),
        grid_(
            std::make_shared<streamer::NavgridMessage>(
                DijkstraPlanner(settings.navgrid_path, settings.spline.turn_penalty,
                                settings.robot_radius_meters).GetGrid())),
        input_(std::move(input)),
        streamer_("/navgrid", settings.navgrid_streamer_port,
                  settings.field_image_path),
        draw_(streamer_.CreateCallback()) {
    streamer_.AddNavgrid(kGrid);
    streamer_.AddNavgrid(kRouteOverlay);
    streamer_.AddPolyline(kSplineOverlay);
    streamer_.AddPolyline(kVelocityOverlay);
  }

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      {
        std::scoped_lock lock(mutex_);
        if (!context->Exists(kPose) || !context->Exists(kInput) ||
            !context->Exists(kSpline) ||
            !context->Exists(kSplineOverlay) || !context->Exists(kCommand) ||
            context->Exists(kGrid))
          return;
        const auto now = std::chrono::steady_clock::now();
        if (now < next_frame_) {
          context->SetMessage(kGrid, nullptr);
          context->SetMessage(kRouteOverlay, nullptr);
          context->SetMessage(kVelocityOverlay, nullptr);
          return;
        }
        next_frame_ = now + std::chrono::milliseconds(100);

        auto route = std::make_shared<streamer::NavgridMessage>(*grid_);
        route->r = route->b = 0;
        route->g = 255;
        for (auto& row : route->navgrid)
          std::ranges::fill(row, false);
        const auto* input = context->GetMessage<InputMessage>(kInput);
        if (input && input->pose_local_us != 0) {
          const double x = input->pose.X().value();
          const double y = input->pose.Y().value();
          if (x >= 0 && x < grid_->field_size_x &&
              y >= 0 && y < grid_->field_size_y) {
            const auto col = static_cast<size_t>(x / grid_->node_size_meters);
            const auto row = static_cast<size_t>(y / grid_->node_size_meters);
            if (row < route->navgrid.size() &&
                col < route->navgrid[row].size()) {
              route->robot_cell =
                  cv::Point(static_cast<int>(col), static_cast<int>(row));
            }
          }
        }
        if (const auto* spline =
                context->GetMessage<SplinePathMessage>(kSpline)) {
          for (const auto& point : spline->raw_route) {
            const double x = point.X().value(), y = point.Y().value();
            if (std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 &&
                x < grid_->field_size_x && y < grid_->field_size_y) {
              const auto col = static_cast<size_t>(x / grid_->node_size_meters);
              const auto row = static_cast<size_t>(y / grid_->node_size_meters);
              if (row < route->navgrid.size() &&
                  col < route->navgrid[row].size())
                route->navgrid[row][col] = true;
            }
          }
        }
        context->SetMessage(kGrid, grid_);
        context->SetMessage(kRouteOverlay, std::move(route));

        auto velocity = std::make_shared<streamer::PolylineMessage>();
        velocity->field_size_x = grid_->field_size_x;
        velocity->field_size_y = grid_->field_size_y;
        velocity->g = velocity->b = 255;
        velocity->arrow_head = true;
        const auto* pose = context->GetMessage<PoseMessage>(kPose);
        const auto* command = context->GetMessage<CommandMessage>(kCommand);
        if (pose && command) {
          const bool following = command->status == PathingStatus::Following &&
                                 command->input &&
                                 input_->ValidateInput(*command->input) ==
                                     PathingStatus::Following;
          const double x = pose->goal_position.X().value();
          const double y = pose->goal_position.Y().value();
          // Same one-second velocity scale used by the standalone visual demo.
          velocity->points = {{x, y},
                              {x + (following ? command->vx : 0),
                               y + (following ? command->vy : 0)}};
        }
        context->SetMessage(kVelocityOverlay, std::move(velocity));
        draw_(context);
      }
      Notify(context);
    };
  }

 private:
  std::shared_ptr<streamer::NavgridMessage> grid_;
  std::shared_ptr<InputNode> input_;
  streamer::NavgridStreamerNode streamer_;
  std::function<void(const control_loop::Context&)> draw_;
  std::chrono::steady_clock::time_point next_frame_{};
};

auto Positive(double value) -> bool {
  return std::isfinite(value) && value > 0;
}

void Validate(const NetworkTablesPathingSettings& settings,
              const Clock& clock) {
  if (settings.navgrid_streamer && (settings.navgrid_streamer_port < 1 ||
                                    settings.navgrid_streamer_port > 65535))
    throw std::invalid_argument(
        "Navgrid streamer port must be between 1 and 65535");
  if (!clock || settings.period.count() <= 0 ||
      settings.command_timeout.count() <= 0 ||
      settings.period > settings.command_timeout ||
      !Positive(settings.lookahead.value()) ||
      !Positive(settings.motion.max_translation.value()) ||
      !std::isfinite(settings.motion.max_rotation.value()) ||
      settings.motion.max_rotation.value() < 0 ||
      !Positive(settings.motion.translation_gain) ||
      !Positive(settings.motion.heading_gain) ||
      !Positive(settings.motion.position_tolerance.value()) ||
      !Positive(settings.motion.heading_tolerance.value()))
    throw std::invalid_argument("Invalid NetworkTables pathing settings");
  if (settings.pose_topic.empty() || settings.goal_topic.empty() ||
      settings.command_topic.empty() || settings.pose_topic.front() != '/' ||
      settings.goal_topic.front() != '/' ||
      settings.command_topic.front() != '/' ||
      settings.pose_topic == settings.goal_topic ||
      settings.command_topic == settings.pose_topic ||
      settings.command_topic == settings.goal_topic)
    throw std::invalid_argument(
        "Pathing NT topics must be absolute and distinct");
}
}  // namespace

class NetworkTablesPathing::Impl {
 public:
  Impl(const NetworkTablesPathingSettings& settings,
       nt::NetworkTableInstance instance, Clock clock)
      : loop(settings.period) {
    Validate(settings, clock);
    auto input = std::make_shared<InputNode>(settings, instance);
    auto spline_settings = settings.spline;
    spline_settings.robot_radius_meters = settings.robot_radius_meters;
    auto spline = std::make_shared<DijkstraSplineNode>(
        kPose, kGoal, kSpline, settings.navgrid_path, spline_settings);
    if (settings.navgrid_streamer)
      spline->EnableSplineDebug(kSplineOverlay);
    auto follower = std::make_shared<FollowerNode>(settings);
    output = std::make_shared<OutputNode>(settings, instance, clock, input);
    loop.SetMaxContext(1);
    loop.RegisterDependencyNode(input);
    loop.RegisterNode(spline);
    loop.RegisterNode(follower);
    loop.RegisterNode(output);
    if (settings.navgrid_streamer)
      loop.RegisterNode(std::make_shared<VisualizationNode>(settings, input));
  }
  control_loop::ControlLoop loop;
  std::shared_ptr<OutputNode> output;
  bool started = false;
};

NetworkTablesPathing::NetworkTablesPathing(
    NetworkTablesPathingSettings settings, nt::NetworkTableInstance instance,
    Clock clock)
    : impl_(std::make_unique<Impl>(settings, instance, std::move(clock))) {}

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
