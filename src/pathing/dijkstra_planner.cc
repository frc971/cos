#include "pathing/dijkstra_planner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace pathing {
namespace {
// Counterclockwise travel directions; adjacent entries differ by 45 degrees.
constexpr std::array<std::pair<int, int>, 8> kDirections{
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
constexpr int kNoDirection = 8;
constexpr int kStatesPerCell = 9;

}  // namespace

DijkstraPlanner::DijkstraPlanner(const std::filesystem::path& navgrid_path,
                                 double turn_penalty, double robot_radius_meters)
    : grid_(navgrid_path), obstacles_(grid_.navgrid),
      turn_penalty_(turn_penalty), robot_radius_(robot_radius_meters) {
  if (!std::isfinite(robot_radius_) || robot_radius_ < 0) {
    throw std::invalid_argument("Robot radius must be finite and nonnegative");
  }
  if (!std::isfinite(turn_penalty) || turn_penalty < 0) {
    throw std::invalid_argument("Turn penalty must be finite and nonnegative");
  }
  if (!std::isfinite(grid_.field_size_x) || grid_.field_size_x <= 0 ||
      !std::isfinite(grid_.field_size_y) || grid_.field_size_y <= 0 ||
      !std::isfinite(grid_.node_size_meters) || grid_.node_size_meters <= 0 ||
      grid_.navgrid.empty() || grid_.navgrid.front().empty()) {
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
          std::numeric_limits<int>::max() / kStatesPerCell) {
    throw std::invalid_argument(
        "Navgrid dimensions do not match field coverage");
  }
  if (robot_radius_ > 0) {
    for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
      for (int x = 0; x < static_cast<int>(width); ++x) {
        const auto center = Center({x, y});
        grid_.navgrid[y][x] = !IsBoxFree(center.X().value(), center.Y().value(),
                                        center.X().value(), center.Y().value());
      }
    }
    // Geometry is immutable. Cache swept-footprint checks once instead of
    // repeating them for every incoming-direction state in every search.
    free_edges_.resize(width * grid_.navgrid.size(), 0);
    for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
      for (int x = 0; x < static_cast<int>(width); ++x) {
        if (!IsFree({x, y}))
          continue;
        const auto a = Center({x, y});
        for (int direction = 0; direction < kNoDirection; ++direction) {
          const auto [dx, dy] = kDirections[direction];
          if (!IsFree({x + dx, y + dy}))
            continue;
          const auto b = Center({x + dx, y + dy});
          if (IsVisible(a.X().value(), a.Y().value(), b.X().value(), b.Y().value()))
            free_edges_[y * width + x] |= uint8_t{1} << direction;
        }
      }
    }
  }
}

auto DijkstraPlanner::IsOnField(const frc::Pose2d& pose) const -> bool {
  return pose.X().value() >= 0 && pose.X().value() < grid_.field_size_x &&
         pose.Y().value() >= 0 && pose.Y().value() < grid_.field_size_y;
}

auto DijkstraPlanner::BlockedStartCells(const frc::Pose2d& pose) const
    -> std::vector<Cell> {
  if (!IsOnField(pose))
    return {};
  if (robot_radius_ == 0) {
    const auto cell = CellFor(pose);
    if (obstacles_[cell.y][cell.x])
      return {cell};
    return {};
  }
  const double px = pose.X().value(), py = pose.Y().value();
  const double size = grid_.node_size_meters;
  std::vector<Cell> cleared;
  for (int y = 0; y < static_cast<int>(obstacles_.size()); ++y) {
    for (int x = 0; x < static_cast<int>(obstacles_[y].size()); ++x) {
      if (!obstacles_[y][x])
        continue;
      const double dx = std::max({x * size - px, px - (x + 1) * size, 0.0});
      const double dy = std::max({y * size - py, py - (y + 1) * size, 0.0});
      if (std::hypot(dx, dy) <= robot_radius_)
        cleared.push_back({x, y});
    }
  }
  return cleared;
}

auto DijkstraPlanner::IsValidStartPose(const frc::Pose2d& pose,
                                       std::span<const Cell> cleared) const
    -> bool {
  return IsOnField(pose) && std::isfinite(pose.Rotation().Radians().value()) &&
         (robot_radius_ == 0 ||
          IsBoxFree(pose.X().value(), pose.Y().value(),
                    pose.X().value(), pose.Y().value(), cleared));
}

auto DijkstraPlanner::IsValidPose(const frc::Pose2d& pose,
                                  std::span<const Cell> cleared) const -> bool {
  return IsOnField(pose) &&
         std::isfinite(pose.Rotation().Radians().value()) &&
         IsFree(CellFor(pose), cleared) &&
         (robot_radius_ == 0 ||
          IsBoxFree(pose.X().value(), pose.Y().value(),
                    pose.X().value(), pose.Y().value(), cleared));
}

auto DijkstraPlanner::CellFor(const frc::Pose2d& pose) const -> Cell {
  return {.x = static_cast<int>(pose.X().value() / grid_.node_size_meters),
          .y = static_cast<int>(pose.Y().value() / grid_.node_size_meters)};
}

auto DijkstraPlanner::Center(Cell cell) const -> frc::Translation2d {
  const double size = grid_.node_size_meters;
  return {units::meter_t{(cell.x * size +
                          std::min((cell.x + 1) * size, grid_.field_size_x)) /
                         2},
          units::meter_t{(cell.y * size +
                          std::min((cell.y + 1) * size, grid_.field_size_y)) /
                         2}};
}

auto DijkstraPlanner::IsBoxFree(double min_x, double min_y, double max_x,
                                double max_y,
                                std::span<const Cell> cleared) const -> bool {
  if (!std::isfinite(min_x) || !std::isfinite(min_y) || !std::isfinite(max_x) ||
      !std::isfinite(max_y) || min_x < robot_radius_ || min_y < robot_radius_ ||
      max_x + robot_radius_ >= grid_.field_size_x ||
      max_y + robot_radius_ >= grid_.field_size_y ||
      min_x > max_x || min_y > max_y) {
    return false;
  }
  const double size = grid_.node_size_meters;
  for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
    for (int x = 0; x < static_cast<int>(grid_.navgrid[y].size()); ++x) {
      if (!obstacles_[y][x] || std::ranges::find(cleared, Cell{x, y}) != cleared.end())
        continue;
      const double dx = std::max({x * size - max_x,
                                  min_x - (x + 1) * size, 0.0});
      const double dy = std::max({y * size - max_y,
                                  min_y - (y + 1) * size, 0.0});
      if (std::hypot(dx, dy) <= robot_radius_) {
        return false;
      }
    }
  }
  return true;
}

auto DijkstraPlanner::IsFree(Cell cell, std::span<const Cell> cleared) const
    -> bool {
  if (cell.x < 0 || cell.y < 0 ||
      cell.x >= static_cast<int>(grid_.navgrid.front().size()) ||
      cell.y >= static_cast<int>(grid_.navgrid.size()))
    return false;
  if (!grid_.navgrid[cell.y][cell.x])
    return true;
  if (cleared.empty())
    return false;
  if (robot_radius_ == 0)
    return std::ranges::find(cleared, cell) != cleared.end();
  const auto center = Center(cell);
  return IsBoxFree(center.X().value(), center.Y().value(),
                   center.X().value(), center.Y().value(), cleared);
}

// Closed rectangle intersection deliberately rejects touching blocked corners.
auto DijkstraPlanner::IsVisible(double x0, double y0, double x1,
                                double y1, std::span<const Cell> cleared) const
    -> bool {
  if (robot_radius_ > 0 &&
      (!IsBoxFree(x0, y0, x0, y0, cleared) ||
       !IsBoxFree(x1, y1, x1, y1, cleared)))
    return false;
  const double size = grid_.node_size_meters;
  for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
    for (int x = 0; x < static_cast<int>(grid_.navgrid[y].size()); ++x) {
      if (!obstacles_[y][x] || std::ranges::find(cleared, Cell{x, y}) != cleared.end())
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
      if (robot_radius_ > 0) {
        // Endpoint-to-box and corner-to-segment distances cover the closest
        // pair between a segment and a disjoint axis-aligned obstacle box.
        const auto endpoint_distance = [&](double px, double py) {
          return std::hypot(std::max({x * size - px, px - (x + 1) * size, 0.0}),
                            std::max({y * size - py, py - (y + 1) * size, 0.0}));
        };
        double distance = std::min(endpoint_distance(x0, y0),
                                   endpoint_distance(x1, y1));
        const double dx = x1 - x0, dy = y1 - y0;
        const double length_squared = dx * dx + dy * dy;
        for (int corner_y = y; corner_y <= y + 1; ++corner_y) {
          for (int corner_x = x; corner_x <= x + 1; ++corner_x) {
            const double px = corner_x * size, py = corner_y * size;
            const double t = length_squared == 0 ? 0 : std::clamp(
                ((px - x0) * dx + (py - y0) * dy) / length_squared, 0.0, 1.0);
            distance = std::min(distance,
                std::hypot(px - (x0 + t * dx), py - (y0 + t * dy)));
          }
        }
        if (distance <= robot_radius_)
          return false;
      }
    }
  }
  return true;
}

auto DijkstraPlanner::FindPath(Cell start, Cell goal,
                                std::span<const Cell> cleared) const
    -> std::vector<Cell> {
  const int width = static_cast<int>(grid_.navgrid.front().size());
  const int count = width * static_cast<int>(grid_.navgrid.size());
  // A cell reached from different directions can have different future costs.
  // Keep independent distances and parents for each incoming direction.
  const int source =
      (start.y * width + start.x) * kStatesPerCell + kNoDirection;
  const int target = goal.y * width + goal.x;
  std::vector<double> distance(count * kStatesPerCell,
                               std::numeric_limits<double>::infinity());
  std::vector<int> parent(count * kStatesPerCell, -1);
  using Entry = std::pair<double, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<>> queue;
  distance[source] = 0;
  queue.emplace(0, source);
  int destination = -1;
  while (!queue.empty()) {
    const auto [cost, state] = queue.top();
    queue.pop();
    if (cost > distance[state])
      continue;
    const int index = state / kStatesPerCell;
    const int incoming = state % kStatesPerCell;
    if (index == target) {
      destination = state;
      break;
    }
    const Cell cell{.x = index % width, .y = index / width};
    for (int direction = 0; direction < kNoDirection; ++direction) {
      const auto [dx, dy] = kDirections[direction];
      const Cell next{.x = cell.x + dx, .y = cell.y + dy};
      if (!IsFree(next, cleared))
        continue;
      if (dx != 0 && dy != 0 &&
          (!IsFree({.x = cell.x + dx, .y = cell.y}, cleared) ||
           !IsFree({.x = cell.x, .y = cell.y + dy}, cleared)))
        continue;
      if (robot_radius_ > 0 &&
          !(free_edges_[index] & (uint8_t{1} << direction))) {
        if (cleared.empty())
          continue;
        // Removing an obstacle can open previously blocked edges.
        const auto a = Center(cell);
        const auto b = Center(next);
        if (!IsVisible(a.X().value(), a.Y().value(),
                       b.X().value(), b.Y().value(), cleared))
          continue;
      }
      double turn_cost = 0;
      if (incoming != kNoDirection) {
        const int difference = std::abs(direction - incoming);
        const int turns = std::min(difference, kNoDirection - difference);
        turn_cost = turn_penalty_ * (turns * std::numbers::pi / 4);
      }
      const int neighbor =
          (next.y * width + next.x) * kStatesPerCell + direction;
      const double next_cost =
          cost + std::hypot(dx, dy) * grid_.node_size_meters + turn_cost;
      if (next_cost < distance[neighbor]) {
        distance[neighbor] = next_cost;
        parent[neighbor] = state;
        queue.emplace(next_cost, neighbor);
      }
    }
  }
  if (destination == -1)
    return {};
  std::vector<Cell> route;
  for (int state = destination; state != -1; state = parent[state]) {
    const int index = state / kStatesPerCell;
    route.push_back({.x = index % width, .y = index / width});
  }
  std::ranges::reverse(route);
  return route;
}

}  // namespace pathing
