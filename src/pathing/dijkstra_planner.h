#pragma once

#include <filesystem>
#include <optional>
#include <cstdint>
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
  // Ignore only the mapped obstacle containing the current robot position.
  auto BlockedStartCell(const frc::Pose2d& pose) const -> std::optional<Cell>;
  auto IsValidPose(const frc::Pose2d& pose,
                   std::optional<Cell> cleared = std::nullopt) const -> bool;
  // CellFor requires a validated pose; FindPath requires free endpoints.
  auto CellFor(const frc::Pose2d& pose) const -> Cell;
  auto Center(Cell cell) const -> frc::Translation2d;
  auto IsFree(Cell cell, std::optional<Cell> cleared = std::nullopt) const -> bool;
  auto IsVisible(double x0, double y0, double x1, double y1,
                 std::optional<Cell> cleared = std::nullopt) const -> bool;
  // A conservative certificate: touching blocked cells is not free.
  auto IsBoxFree(double min_x, double min_y, double max_x, double max_y,
                 std::optional<Cell> cleared = std::nullopt) const -> bool;
  auto FindPath(Cell start, Cell goal,
                std::optional<Cell> cleared = std::nullopt) const -> std::vector<Cell>;
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
