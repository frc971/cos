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
                                 double turn_penalty)
    : grid_(navgrid_path), turn_penalty_(turn_penalty) {
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
}

auto DijkstraPlanner::IsValidPose(const frc::Pose2d& pose) const -> bool {
  return std::isfinite(pose.X().value()) && std::isfinite(pose.Y().value()) &&
         std::isfinite(pose.Rotation().Radians().value()) &&
         pose.X().value() >= 0 && pose.X().value() < grid_.field_size_x &&
         pose.Y().value() >= 0 && pose.Y().value() < grid_.field_size_y &&
         IsFree(CellFor(pose));
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
                                double max_y) const -> bool {
  if (!std::isfinite(min_x) || !std::isfinite(min_y) || !std::isfinite(max_x) ||
      !std::isfinite(max_y) || min_x < 0 || min_y < 0 ||
      max_x >= grid_.field_size_x || max_y >= grid_.field_size_y ||
      min_x > max_x || min_y > max_y) {
    return false;
  }
  const double size = grid_.node_size_meters;
  for (int y = 0; y < static_cast<int>(grid_.navgrid.size()); ++y) {
    for (int x = 0; x < static_cast<int>(grid_.navgrid[y].size()); ++x) {
      if (grid_.navgrid[y][x] && min_x <= (x + 1) * size && max_x >= x * size &&
          min_y <= (y + 1) * size && max_y >= y * size) {
        return false;
      }
    }
  }
  return true;
}

auto DijkstraPlanner::IsFree(Cell cell) const -> bool {
  return cell.y >= 0 && cell.x >= 0 &&
         cell.y < static_cast<int>(grid_.navgrid.size()) &&
         cell.x < static_cast<int>(grid_.navgrid.front().size()) &&
         !grid_.navgrid[cell.y][cell.x];
}

// Closed rectangle intersection deliberately rejects touching blocked corners.
auto DijkstraPlanner::IsVisible(double x0, double y0, double x1,
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

auto DijkstraPlanner::FindPath(Cell start, Cell goal) const
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
      if (!IsFree(next))
        continue;
      if (dx != 0 && dy != 0 &&
          (!IsFree({.x = cell.x + dx, .y = cell.y}) ||
           !IsFree({.x = cell.x, .y = cell.y + dy})))
        continue;
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
