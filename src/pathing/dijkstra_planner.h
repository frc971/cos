#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <frc/geometry/Pose2d.h>

#include "streamer/navgrid_streamer_node.h"

namespace pathing {

// Shared, immutable navgrid search and collision geometry for pathing nodes.
class DijkstraPlanner {
 public:
  struct Cell {
    int x;
    int y;
    auto operator==(const Cell&) const -> bool = default;
  };

  explicit DijkstraPlanner(const std::filesystem::path& navgrid_path,
                           double turn_penalty = 0.3,
                           double robot_radius_meters = 0);
  // Ignore mapped obstacles overlapping the current robot footprint.
  auto BlockedStartCells(const frc::Pose2d& pose) const -> std::vector<Cell>;
  auto IsValidStartPose(const frc::Pose2d& pose,
                        std::span<const Cell> cleared) const -> bool;
  auto IsValidPose(const frc::Pose2d& pose,
                   std::span<const Cell> cleared = {}) const -> bool;
  // CellFor requires an in-bounds pose.
  auto CellFor(const frc::Pose2d& pose) const -> Cell;
  auto Center(Cell cell) const -> frc::Translation2d;
  auto IsFree(Cell cell, std::span<const Cell> cleared = {}) const -> bool;
  auto IsVisible(double x0, double y0, double x1, double y1,
                 std::span<const Cell> cleared = {}) const -> bool;
  // A conservative certificate: touching blocked cells is not free.
  auto IsBoxFree(double min_x, double min_y, double max_x, double max_y,
                 std::span<const Cell> cleared = {}) const -> bool;
  // Requires in-bounds endpoints and a free goal.
  auto FindPath(Cell start, Cell goal,
                std::span<const Cell> cleared = {}) const -> std::vector<Cell>;
  auto GetGrid() const -> const streamer::NavgridMessage& { return grid_; }

 private:
  auto IsOnField(const frc::Pose2d& pose) const -> bool;
  streamer::NavgridMessage grid_;
  std::vector<std::vector<bool>> obstacles_;
  std::vector<uint8_t> free_edges_;
  double turn_penalty_;
  double robot_radius_;
};

}  // namespace pathing
