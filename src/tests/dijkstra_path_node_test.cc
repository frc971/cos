#include "pathing/dijkstra_path_node.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

class DijkstraPathNodeTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    grid_path_ = std::filesystem::temp_directory_path() /
                 ("dijkstra_test_" + std::to_string(timestamp) + ".json");
    // Rows run from low y to high y. The upper corridor is shorter but has
    // six 90-degree turns; the lower corridor is longer with only two turns.
    WriteGrid({".......", ".#####.", ".#####.", ".#...#.", "...#..."});
  }

  void TearDown() override { std::filesystem::remove(grid_path_); }

  void WriteGrid(const std::vector<std::string>& rows, double node_size = 1.0) {
    std::vector<std::vector<bool>> grid;
    for (const auto& row : rows) {
      auto& cells = grid.emplace_back();
      for (char value : row) {
        cells.push_back(value == '#');
      }
    }
    std::ofstream file(grid_path_);
    file << nlohmann::json{
        {"field_size", {{"x", rows.front().size() * node_size},
                        {"y", rows.size() * node_size}}},
        {"nodeSizeMeters", node_size}, {"grid", grid}};
  }

  auto Plan(double penalty, double goal_x = 6.5, double goal_y = 3.5)
      -> control_loop::Context {
    pathing::MotionSettings settings;
    settings.turn_penalty = penalty;
    pathing::DijkstraPathNode planner("pose", "goal", "velocity", grid_path_,
                                     settings);
    planner.EnableNavgridDebug("route");
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
    context->SetMessage("pose", std::make_shared<pathing::PoseMessage>(
        frc::Pose2d{units::meter_t{0.5}, units::meter_t{3.5},
                    frc::Rotation2d{}}));
    context->SetMessage("goal", std::make_shared<pathing::PoseMessage>(
        frc::Pose2d{units::meter_t{goal_x}, units::meter_t{goal_y},
                    frc::Rotation2d{}}));
    planner.CreateCallback()(context);
    return context;
  }

  std::filesystem::path grid_path_;
};

TEST_F(DijkstraPathNodeTest, ZeroPenaltySelectsShorterWindingCorridor) {
  auto context = Plan(0);
  const auto* output = context->GetMessage<pathing::VelocityGoalMessage>("velocity");
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->status, pathing::PathStatus::FollowingPath);
  const auto* route = context->GetMessage<streamer::NavgridMessage>("route");
  ASSERT_NE(route, nullptr);
  EXPECT_TRUE(route->navgrid[4][1]);
  EXPECT_FALSE(route->navgrid[0][3]);
}

TEST_F(DijkstraPathNodeTest, RobotRadiusChecksPosesSweptSegmentsAndSplineBoxes) {
  WriteGrid({".......", ".......", "..#....", ".......", ".......", ".......", "......."});
  pathing::DijkstraPlanner point_robot(grid_path_, 0, 0);
  pathing::DijkstraPlanner robot(grid_path_, 0, 0.38);
  const auto pose = [](double x, double y) {
    return frc::Pose2d{units::meter_t{x}, units::meter_t{y}, frc::Rotation2d{}};
  };
  EXPECT_TRUE(point_robot.IsValidPose(pose(1.7, 2.5)));
  EXPECT_FALSE(robot.IsValidPose(pose(1.7, 2.5)));
  EXPECT_TRUE(robot.IsValidPose(pose(1.5, 2.5)));
  EXPECT_FALSE(robot.IsValidPose(pose(0.2, 1.5)));
  EXPECT_FALSE(robot.IsValidPose(pose(6.8, 1.5)));
  EXPECT_TRUE(robot.IsValidPose(pose(0.5, 1.5)));
  EXPECT_TRUE(point_robot.IsVisible(1.7, 1.5, 1.7, 3.5));
  EXPECT_FALSE(robot.IsVisible(1.7, 1.5, 1.7, 3.5));
  EXPECT_TRUE(robot.IsVisible(1.5, 1.5, 1.5, 3.5));
  // A circular footprint can clear a corner diagonally even when its square
  // bounding box overlaps the obstacle.
  EXPECT_TRUE(robot.IsVisible(1.7, 1.7, 1.7, 1.7));
  EXPECT_FALSE(robot.IsBoxFree(1.0, 1.5, 1.7, 3.5));
  EXPECT_TRUE(robot.IsBoxFree(1.0, 1.5, 1.5, 3.5));
}

TEST_F(DijkstraPathNodeTest, RobotDiameterCannotFitThroughNarrowCorridor) {
  WriteGrid(std::vector<std::string>(7, ".#...#."), 0.2);
  pathing::DijkstraPlanner small(grid_path_, 0, 0.28);
  pathing::DijkstraPlanner robot(grid_path_, 0, 0.38);
  EXPECT_TRUE(small.IsFree({3, 3}));
  EXPECT_FALSE(robot.IsFree({3, 3}));
  EXPECT_FALSE(small.FindPath({3, 2}, {3, 4}).empty());
}

TEST_F(DijkstraPathNodeTest, RejectsNegativeAndNonfiniteRobotRadii) {
  for (double radius : {-0.1, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW((pathing::DijkstraPlanner(grid_path_, 0, radius)),
                 std::invalid_argument);
  }
}

TEST_F(DijkstraPathNodeTest, TurnPenaltySelectsLongerStraighterCorridor) {
  auto context = Plan(1);
  const auto* output = context->GetMessage<pathing::VelocityGoalMessage>("velocity");
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->status, pathing::PathStatus::FollowingPath);
  const auto* route = context->GetMessage<streamer::NavgridMessage>("route");
  ASSERT_NE(route, nullptr);
  EXPECT_TRUE(route->navgrid[0][3]);
  EXPECT_FALSE(route->navgrid[4][1]);
}

TEST_F(DijkstraPathNodeTest, SameCellNeedsNoIncomingDirection) {
  auto context = Plan(1, 0.5, 3.5);
  const auto* output = context->GetMessage<pathing::VelocityGoalMessage>("velocity");
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->status, pathing::PathStatus::ReachedGoal);
  EXPECT_EQ(output->vx.value(), 0);
  EXPECT_EQ(output->vy.value(), 0);
}

TEST_F(DijkstraPathNodeTest, UnreachableGoalStillReportsNoPath) {
  WriteGrid({"#######", "#######", "#######", ".#####.", "#######"});
  auto context = Plan(1);
  const auto* output = context->GetMessage<pathing::VelocityGoalMessage>("velocity");
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->status, pathing::PathStatus::NoPath);
}

TEST_F(DijkstraPathNodeTest, RejectsNegativeAndNonfinitePenalties) {
  for (double penalty : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    pathing::MotionSettings settings;
    settings.turn_penalty = penalty;
    EXPECT_THROW((pathing::DijkstraPathNode{
        "pose", "goal", "velocity", grid_path_, settings}), std::invalid_argument);
  }
}

}  // namespace
