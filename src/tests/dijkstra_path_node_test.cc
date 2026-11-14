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

  void WriteGrid(const std::vector<std::string>& rows) {
    std::vector<std::vector<bool>> grid;
    for (const auto& row : rows) {
      auto& cells = grid.emplace_back();
      for (char value : row) {
        cells.push_back(value == '#');
      }
    }
    std::ofstream file(grid_path_);
    file << nlohmann::json{
        {"field_size", {{"x", rows.front().size()}, {"y", rows.size()}}},
        {"nodeSizeMeters", 1.0}, {"grid", grid}};
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
