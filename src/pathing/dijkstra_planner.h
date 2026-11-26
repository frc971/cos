#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <frc/geometry/Pose2d.h>

#include "streamer/navgrid_streamer_node.h"

namespace pathing {

// Shared, immutable navgrid search and collision geometry for pathing nodes.
class DijkstraPlanner {
 public:
  // Zero-based column and row measured from the lower-left field corner.
  struct Cell {
    int x;
    int y;
    auto operator==(const Cell&) const -> bool = default;
  };

  explicit DijkstraPlanner(const std::filesystem::path& navgrid_path,
                           double turn_penalty = 0.3,
                           double robot_radius_meters = 0);
  // Returns occupied cells overlapping the measured footprint. Pass these as
  // cleared to ignore them for a single plan without changing the map.
  auto BlockedStartCells(const frc::Pose2d& pose) const -> std::vector<Cell>;
  // Checks the measured start position and heading, including footprint clearance.
  auto IsValidStartPose(const frc::Pose2d& pose,
                        std::span<const Cell> cleared) const -> bool;
  // Checks the exact pose and the center of its containing cell.
  auto IsValidPose(const frc::Pose2d& pose,
                   std::span<const Cell> cleared = {}) const -> bool;
  // CellFor requires an in-bounds pose.
  auto CellFor(const frc::Pose2d& pose) const -> Cell;
  auto Center(Cell cell) const -> frc::Translation2d;
  // Tests footprint clearance at the cell center.
  auto IsFree(Cell cell, std::span<const Cell> cleared = {}) const -> bool;
  // Tests whether the footprint can traverse a straight segment in field meters.
  auto IsVisible(double x0, double y0, double x1, double y1,
                 std::span<const Cell> cleared = {}) const -> bool;
  // A conservative certificate: touching blocked cells is not free.
  auto IsBoxFree(double min_x, double min_y, double max_x, double max_y,
                 std::span<const Cell> cleared = {}) const -> bool;
  // Returns cells from start through goal, or an empty vector if no route exists.
  // Requires in-bounds endpoints and a free goal. An optional measured start
  // position replaces the start cell center for the first edge.
  auto FindPath(Cell start, Cell goal,
                std::span<const Cell> cleared = {},
                std::optional<frc::Translation2d> start_position = std::nullopt)
      const -> std::vector<Cell>;
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
