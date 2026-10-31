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

TEST(BallPathingLogicTest, MaintainsMinimumSpeedNearTargetAndLimitsMaximum) {
  const cv::Point2d near = ComputeVelocity({0.2, 0.0}, 0.75, 1.5, 1.0, 1.5);
  EXPECT_NEAR(near.x, 1.0, 1e-12);
  EXPECT_NEAR(near.y, 0.0, 1e-12);
  const cv::Point2d proportional =
      ComputeVelocity({1.2, 0.4}, 0.75, 1.5, 1.0, 1.5);
  EXPECT_NEAR(proportional.x, 0.9, 1e-12);
  EXPECT_NEAR(proportional.y, 0.6, 1e-12);
  const cv::Point2d limited =
      ComputeVelocity({3.0, 4.0}, 0.75, 1.5, 1.0, 1.5);
  EXPECT_NEAR(cv::norm(limited), 1.5, 1e-12);
  EXPECT_GT(limited.x, 0.0);
  EXPECT_GT(limited.y, 0.0);
}

TEST(BallPathingLogicTest, ZeroTargetDoesNotProduceUndefinedDirection) {
  EXPECT_EQ(ComputeVelocity({0.0, 0.0}, 0.75, 1.5, 1.0, 1.5), cv::Point2d());
}

TEST(BallPathingLogicTest, CanDisableMinimumSpeed) {
  const auto velocity = ComputeVelocity({0.2, 0.0}, 0.75, 1.5, 0.0, 1.5);
  EXPECT_NEAR(velocity.x, 0.15, 1e-12);
  EXPECT_NEAR(velocity.y, 0.0, 1e-12);
}

TEST(BallPathingLogicTest, PreservesLateralBiasAtMinimumAndMaximumSpeed) {
  for (const double offset : {0.4, 2.0}) {
    const cv::Point2d velocity =
        ComputeVelocity({offset, offset}, 0.75, 1.5, 1.0, 1.5);
    EXPECT_NEAR(velocity.y, 2.0 * velocity.x, 1e-12);
    if (offset == 0.4) {
      EXPECT_NEAR(cv::norm(velocity), 1.0, 1e-12);
    } else {
      EXPECT_NEAR(cv::norm(velocity), 1.5, 1e-12);
    }
  }
}

TEST(BallPathingLogicTest, PreservesForwardAndLateralSigns) {
  for (const double x : {-2.0, 2.0}) {
    for (const double y : {-2.0, 2.0}) {
      const cv::Point2d velocity =
          ComputeVelocity({x, y}, 0.75, 1.5, 1.0, 1.5);
      EXPECT_GT(velocity.x * x, 0.0);
      EXPECT_GT(velocity.y * y, 0.0);
      EXPECT_NEAR(cv::norm(velocity), 1.5, 1e-12);
    }
  }
}

TEST(BallPathingLogicTest, CommandsNegativeXForTargetBehindRobot) {
  const cv::Point2d velocity =
      ComputeVelocity({-2.0, 0.5}, 0.75, 1.5, 1.0, 1.5);
  EXPECT_NEAR(cv::norm(velocity), 1.5, 1e-12);
  EXPECT_LT(velocity.x, 0.0);
  EXPECT_GT(velocity.y, 0.0);
}

}  // namespace
}  // namespace gamepiece::internal
