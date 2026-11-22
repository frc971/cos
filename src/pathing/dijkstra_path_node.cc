#include "pathing/dijkstra_path_node.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace pathing {
namespace {
auto Positive(double value) -> bool {
  return std::isfinite(value) && value > 0;
}
}  // namespace

DijkstraPathNode::DijkstraPathNode(std::string_view pose_path,
                                   std::string_view goal_path,
                                   std::string_view output_path,
                                   const std::filesystem::path& navgrid_path,
                                   MotionSettings settings)
    : pose_path_(pose_path),
      goal_path_(goal_path),
      output_path_(output_path),
      planner_(navgrid_path, settings.turn_penalty, settings.robot_radius_meters),
      settings_(settings),
      dependencies_({{pose_path_, typeid(PoseMessage)},
                     {goal_path_, typeid(PoseMessage)}}),
      publications_({{output_path_, typeid(VelocityGoalMessage)}}) {
  if (pose_path_.empty() || goal_path_.empty() || output_path_.empty() ||
      pose_path_ == goal_path_ || output_path_ == pose_path_ ||
      output_path_ == goal_path_) {
    throw std::invalid_argument(
        "Planner channels must be nonempty and distinct");
  }
  if (!Positive(settings.max_translation.value()) ||
      !Positive(settings.max_rotation.value()) ||
      !Positive(settings.translation_gain) ||
      !Positive(settings.heading_gain) ||
      !Positive(settings.position_tolerance.value()) ||
      !Positive(settings.heading_tolerance.value())) {
    throw std::invalid_argument(
        "Planner motion settings must be finite and positive");
  }
}

void DijkstraPathNode::EnableNavgridDebug(std::string_view output_path) {
  std::scoped_lock lock(mutex_);
  if (output_path.empty() || output_path == pose_path_ ||
      output_path == goal_path_ || output_path == output_path_) {
    throw std::invalid_argument("Debug channel must be nonempty and distinct");
  }
  if (!debug_path_.empty()) {
    if (debug_path_ == output_path)
      return;
    throw std::logic_error("Navgrid debug channel is already configured");
  }
  debug_path_ = output_path;
  publications_.emplace_back(debug_path_, typeid(streamer::NavgridMessage));
}

auto DijkstraPathNode::Calculate(const PoseMessage* pose_message,
                                 const PoseMessage* goal_message,
                                 std::vector<Cell>& route) const
    -> VelocityGoalMessage {
  VelocityGoalMessage output;
  if (!pose_message || !goal_message)
    return output;
  const auto& pose = pose_message->goal_position;
  const auto& goal = goal_message->goal_position;
  if (!planner_.IsValidPose(pose) || !planner_.IsValidPose(goal))
    return output;
  const auto& grid = planner_.GetGrid();
  const double size = grid.node_size_meters;
  route = planner_.FindPath(planner_.CellFor(pose), planner_.CellFor(goal));
  output.status = PathStatus::NoPath;
  if (route.empty())
    return output;
  const double x = pose.X().value(), y = pose.Y().value();
  double target_x = goal.X().value(), target_y = goal.Y().value();
  if (!planner_.IsVisible(x, y, target_x, target_y)) {
    bool found = false;
    for (auto& it : std::views::reverse(route)) {
      target_x =
          (it.x * size + std::min((it.x + 1) * size, grid.field_size_x)) / 2;
      target_y =
          (it.y * size + std::min((it.y + 1) * size, grid.field_size_y)) / 2;
      if (planner_.IsVisible(x, y, target_x, target_y)) {
        found = true;
        break;
      }
    }
    if (!found) {
      route.clear();
      return output;
    }
  }
  const bool at_position =
      std::hypot(goal.X().value() - x, goal.Y().value() - y) <=
      settings_.position_tolerance.value();
  const double error = std::remainder(
      goal.Rotation().Radians().value() - pose.Rotation().Radians().value(),
      2 * std::numbers::pi);
  const bool at_heading =
      std::abs(error) <= settings_.heading_tolerance.value();
  output.status = at_position && at_heading ? PathStatus::ReachedGoal
                                            : PathStatus::FollowingPath;
  if (!at_position) {
    const double dx = target_x - x, dy = target_y - y;
    const double distance = std::hypot(dx, dy);
    const double gain =
        distance == 0 ? 0
                      : std::min(settings_.translation_gain,
                                 settings_.max_translation.value() / distance);
    output.vx = units::meters_per_second_t{gain * dx};
    output.vy = units::meters_per_second_t{gain * dy};
  }
  if (!at_heading) {
    output.omega = units::radians_per_second_t{std::clamp(
        settings_.heading_gain * error, -settings_.max_rotation.value(),
        settings_.max_rotation.value())};
  }
  return output;
}

auto DijkstraPathNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    {
      std::scoped_lock lock(mutex_);
      if (!context->Exists(pose_path_) || !context->Exists(goal_path_) ||
          context->Exists(output_path_))
        return;
      std::vector<Cell> route;
      auto output = std::make_shared<VelocityGoalMessage>(
          Calculate(context->GetMessage<PoseMessage>(pose_path_),
                    context->GetMessage<PoseMessage>(goal_path_), route));
      if (!debug_path_.empty()) {
        auto debug =
            std::make_shared<streamer::NavgridMessage>(planner_.GetGrid());
        debug->r = 0;
        debug->g = 255;
        debug->b = 0;
        for (auto& row : debug->navgrid)
          std::ranges::fill(row, false);
        for (const Cell cell : route)
          debug->navgrid[cell.y][cell.x] = true;
        context->SetMessage(debug_path_, std::move(debug));
      }
      context->SetMessage(output_path_, std::move(output));
    }
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

auto DijkstraPathNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}
auto DijkstraPathNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}
void DijkstraPathNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}
}  // namespace pathing
