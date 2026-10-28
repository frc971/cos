#include "gamepiece/ball_pathing_logic.h"

#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace gamepiece::internal {
namespace {

TEST(BallPathingLogicTest, SelectsNeighborhoodWithMostBalls) {
  const std::vector<ProjectedMaskSample> samples{
      {{1.0, 0.0}, 1.0}, {{1.1, 0.1}, 1.0}, {{0.9, -0.1}, 1.0},
      {{4.0, 0.0}, 1.0}, {{4.1, 0.0}, 1.0}};
  const auto clusters = BuildBallClusters(samples, 0.5, 0.1);
  const auto selected = ChooseBallCluster(clusters, std::nullopt, 0.5, 1.25);
  ASSERT_TRUE(selected.has_value());
  EXPECT_DOUBLE_EQ(selected->evidence_m2, 3.0);
  EXPECT_NEAR(selected->centroid.x, 1.0, 1e-12);
  EXPECT_NEAR(selected->centroid.y, 0.0, 1e-12);
}

TEST(BallPathingLogicTest, TieBreaksTowardNearestCluster) {
  const std::vector<BallCluster> clusters{
      {.centroid = {4.0, 0.0}, .evidence_m2 = 3.0},
      {.centroid = {2.0, 0.0}, .evidence_m2 = 3.0}};
  const auto selected = ChooseBallCluster(clusters, std::nullopt, 1.0, 1.25);
  ASSERT_TRUE(selected.has_value());
  EXPECT_DOUBLE_EQ(selected->centroid.x, 2.0);
}

TEST(BallPathingLogicTest, KeepsTrackedClusterUntilMarginIsMet) {
  const BallCluster tracked{.centroid = {2.0, 0.0}, .evidence_m2 = 3.0};
  const std::vector<BallCluster> one_more{
      {.centroid = {2.1, 0.0}, .evidence_m2 = 3.0},
      {.centroid = {5.0, 0.0}, .evidence_m2 = 3.5}};
  auto selected = ChooseBallCluster(one_more, tracked, 0.5, 1.25);
  ASSERT_TRUE(selected.has_value());
  EXPECT_DOUBLE_EQ(selected->centroid.x, 2.1);

  const std::vector<BallCluster> two_more{
      {.centroid = {2.1, 0.0}, .evidence_m2 = 3.0},
      {.centroid = {5.0, 0.0}, .evidence_m2 = 4.0}};
  selected = ChooseBallCluster(two_more, tracked, 0.5, 1.25);
  ASSERT_TRUE(selected.has_value());
  EXPECT_DOUBLE_EQ(selected->centroid.x, 5.0);
}

TEST(BallPathingLogicTest, DrivesStraightForwardAtConstantSpeed) {
  const BallDriveVelocity centered =
      ComputeDriveVelocity({0.2, 0.0}, 2.5, 3.0, 5.0);
  EXPECT_DOUBLE_EQ(centered.vx_mps, 2.5);
  EXPECT_DOUBLE_EQ(centered.vy_mps, 0.0);
  EXPECT_DOUBLE_EQ(centered.omega_rad_per_s, 0.0);

  const BallDriveVelocity behind =
      ComputeDriveVelocity({-2.0, 0.0}, 2.5, 3.0, 5.0);
  EXPECT_DOUBLE_EQ(behind.vx_mps, 2.5);
  EXPECT_DOUBLE_EQ(behind.vy_mps, 0.0);
  EXPECT_DOUBLE_EQ(behind.omega_rad_per_s, 5.0);
}

TEST(BallPathingLogicTest, CommandsProportionalLimitedYawTowardTarget) {
  const BallDriveVelocity left =
      ComputeDriveVelocity({1.0, 0.2}, 2.5, 3.0, 5.0);
  const BallDriveVelocity right =
      ComputeDriveVelocity({1.0, -0.2}, 2.5, 3.0, 5.0);
  EXPECT_NEAR(left.omega_rad_per_s, 3.0 * std::atan2(0.2, 1.0),
              1e-12);
  EXPECT_NEAR(right.omega_rad_per_s, -left.omega_rad_per_s, 1e-12);

  const BallDriveVelocity saturated_left =
      ComputeDriveVelocity({0.0, 1.0}, 2.5, 3.0, 4.0);
  const BallDriveVelocity saturated_right =
      ComputeDriveVelocity({0.0, -1.0}, 2.5, 3.0, 4.0);
  EXPECT_DOUBLE_EQ(saturated_left.omega_rad_per_s, 4.0);
  EXPECT_DOUBLE_EQ(saturated_right.omega_rad_per_s, -4.0);
}

}  // namespace
}  // namespace gamepiece::internal
