#include "pathing/dijkstra_path_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
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
      grid_(navgrid_path),
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
  if (!Positive(grid_.field_size_x) || !Positive(grid_.field_size_y) ||
      !Positive(grid_.node_size_meters) || grid_.navgrid.empty() ||
      grid_.navgrid.front().empty()) {
    throw std::invalid_argument("Invalid navgrid dimensions");
  }
  const auto width = grid_.navgrid.front().size();
  for (const auto& row : grid_.navgrid) {
    if (row.size() != width) {
      throw std::invalid_argument("Navgrid must be rectangular");
    }
  }
  if (width != std::ceil(grid_.field_size_x / grid_.node_size_meters) ||
      grid_.navgrid.size() !=
          std::ceil(grid_.field_size_y / grid_.node_size_meters) ||
      width * static_cast<double>(grid_.navgrid.size()) >
          std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "Navgrid dimensions do not match field coverage");
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

auto DijkstraPathNode::IsFree(Cell cell) const -> bool {
  return cell.y >= 0 && cell.x >= 0 &&
         cell.y < static_cast<int>(grid_.navgrid.size()) &&
         cell.x < static_cast<int>(grid_.navgrid.front().size()) &&
         !grid_.navgrid[cell.y][cell.x];
}

// Closed rectangle intersection deliberately rejects touching blocked corners.
auto DijkstraPathNode::IsVisible(double x0, double y0, double x1,
                                 double y1) const -> bool {
  const double size = grid_.node_size_meters;
  for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
    for (int x = 0; x < static_cast<int>(grid_.navgrid[y].size()); ++x) {
      if (!grid_.navgrid[y][x])
        continue;
      double lo = 0, hi = 1;
      const auto clip = [&](double origin, double delta, double minimum,
                            double maximum) -> bool {
        if (delta == 0)
          return origin >= minimum && origin <= maximum;
        double a = (minimum - origin) / delta;
        double b = (maximum - origin) / delta;
        if (a > b)
          std::swap(a, b);
        lo = std::max(lo, a);
        hi = std::min(hi, b);
        return lo <= hi;
      };
      if (clip(x0, x1 - x0, x * size, (x + 1) * size) &&
          clip(y0, y1 - y0, y * size, (y + 1) * size))
        return false;
    }
  }
  return true;
}

auto DijkstraPathNode::FindPath(Cell start, Cell goal) const
    -> std::vector<Cell> {
  const int width = static_cast<int>(grid_.navgrid.front().size());
  const int count = width * static_cast<int>(grid_.navgrid.size());
  const int source = start.y * width + start.x;
  const int target = goal.y * width + goal.x;
  std::vector<double> distance(count, std::numeric_limits<double>::infinity());
  std::vector<int> parent(count, -1);
  using Entry = std::pair<double, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue;
  distance[source] = 0;
  queue.emplace(0, source);
  while (!queue.empty()) {
    const auto [cost, index] = queue.top();
    queue.pop();
    if (cost > distance[index])
      continue;
    if (index == target)
      break;
    const Cell cell{.x = index % width, .y = index / width};
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const Cell next{.x = cell.x + dx, .y = cell.y + dy};
        if ((dx == 0 && dy == 0) || !IsFree(next))
          continue;
        if (dx != 0 && dy != 0 &&
            (!IsFree({.x = cell.x + dx, .y = cell.y}) ||
             !IsFree({.x = cell.x, .y = cell.y + dy})))
          continue;
        const int neighbor = next.y * width + next.x;
        const double next_cost =
            cost + std::hypot(dx, dy) * grid_.node_size_meters;
        if (next_cost < distance[neighbor]) {
          distance[neighbor] = next_cost;
          parent[neighbor] = index;
          queue.emplace(next_cost, neighbor);
        }
      }
    }
  }
  if (!std::isfinite(distance[target]))
    return {};
  std::vector<Cell> route;
  for (int index = target; index != -1; index = parent[index]) {
    route.push_back({.x = index % width, .y = index / width});
  }
  std::ranges::reverse(route);
  return route;
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
  const auto valid = [&](const frc::Pose2d& value) -> bool {
    return std::isfinite(value.X().value()) &&
           std::isfinite(value.Y().value()) &&
           std::isfinite(value.Rotation().Radians().value()) &&
           value.X().value() >= 0 && value.X().value() < grid_.field_size_x &&
           value.Y().value() >= 0 && value.Y().value() < grid_.field_size_y;
  };
  if (!valid(pose) || !valid(goal))
    return output;
  const double size = grid_.node_size_meters;
  const auto cell = [size](const frc::Pose2d& value) -> Cell {
    return {.x = static_cast<int>(value.X().value() / size),
            .y = static_cast<int>(value.Y().value() / size)};
  };
  if (!IsFree(cell(pose)) || !IsFree(cell(goal)))
    return output;
  route = FindPath(cell(pose), cell(goal));
  output.status = PathStatus::NoPath;
  if (route.empty())
    return output;
  const double x = pose.X().value(), y = pose.Y().value();
  double target_x = goal.X().value(), target_y = goal.Y().value();
  if (!IsVisible(x, y, target_x, target_y)) {
    bool found = false;
    for (auto& it : std::views::reverse(route)) {
      target_x =
          (it.x * size + std::min((it.x + 1) * size, grid_.field_size_x)) / 2;
      target_y =
          (it.y * size + std::min((it.y + 1) * size, grid_.field_size_y)) / 2;
      if (IsVisible(x, y, target_x, target_y)) {
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
        auto debug = std::make_shared<streamer::NavgridMessage>(grid_);
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
