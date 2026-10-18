#include "gamepiece/hsv_cluster_tracker_node.h"

#include <cmath>
#include <memory>
#include <random>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include "localization/position.h"
#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {
struct HSVClusterTrackerNodeTestPeer {
  static auto Run(HSVClusterTrackerNode& tracker, const std::vector<cv::Point2f>& points,
                  std::optional<frc::Pose2d> pose = std::nullopt) {
    tracker.robot_pose_ = pose;
    tracker.ground_points_ = points;
    tracker.ground_weights_.assign(points.size(), 0.1f);
    tracker.refined_weights_ = tracker.ground_weights_;
    tracker.thresholded_points_.clear();
    for (const auto& p : points) tracker.thresholded_points_.push_back({-p.y / p.x, 0.925f / p.x});
    tracker.clusters_ = tracker.ClusterGroundPoints();
    tracker.AssignPersistentIds();
    return *tracker.GetClusters();
  }
};
}
namespace {
const cv::Matx33d kIntrinsics{100, 0, 64, 0, 100, 48, 0, 0, 1};
const cv::Matx44d kExtrinsics{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
using Tracker = gamepiece::HSVClusterTrackerNode;
auto MakeTracker() { return Tracker("decoded", "clusters", kIntrinsics, kExtrinsics, {}, 128 * 96); }
auto Cloud(cv::Point2f center, float sx = 0.35f, float sy = 0.2f) {
  std::mt19937 rng(7);
  std::normal_distribution<float> n;
  std::vector<cv::Point2f> points;
  for (int i = 0; i < 300; ++i) points.push_back(center + cv::Point2f{sx * n(rng), sy * n(rng)});
  return points;
}
auto Cluster(Tracker& tracker, const std::vector<cv::Point2f>& points,
         std::optional<frc::Pose2d> pose = std::nullopt) {
  return gamepiece::HSVClusterTrackerNodeTestPeer::Run(tracker, points, pose);
}
// Ray-cast equal-radius spheres, so projection tests use exact simulated geometry.
const cv::Matx33d kSphereIntrinsics{400, 0, 320, 0, 400, 100, 0, 0, 1};
auto SphereImage(float forward = 4.0f) -> cv::Mat {
  cv::Mat3b hsv(480, 640, cv::Vec3b{0, 0, 0});
  for (int y = 0; y < hsv.rows; ++y) for (int x = 0; x < hsv.cols; ++x) {
    const cv::Vec3f ray{(x - 320) / 400.0f, (y - 100) / 400.0f, 1};
    for (float lateral : {-0.45f, -0.27f, -0.09f, 0.09f, 0.27f, 0.45f}) {
      const cv::Vec3f center{-lateral, 0.925f, forward};
      const float dot = ray.dot(center);
      if (dot > 0 && dot * dot >= ray.dot(ray) * (center.dot(center) - 0.075f * 0.075f))
        hsv(y, x) = {24, 125, 230};
    }
  }
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  return bgr;
}
TEST(HSVClusterTrackerTest, SphereProjectionPreservesLocationAndRejectsAllianceZone) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, {}, 640 * 480);
  const auto image = SphereImage();
  tracker.ProcessFrame(image, frc::Pose2d{2_m, 4_m, frc::Rotation2d{0_rad}});
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  const auto first = tracker.GetClusters()->front();
  EXPECT_NEAR(first.robot_relative_translation->X().value(), 4, 0.08);
  EXPECT_NEAR(first.robot_relative_translation->Y().value(), 0, 0.03);
  EXPECT_LT(first.ground_covar(0, 0), 0.02);
  tracker.ProcessFrame(SphereImage(3), frc::Pose2d{3_m, 4_m, frc::Rotation2d{0_rad}});
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(first.id, tracker.GetClusters()->front().id);
  tracker.ProcessFrame(image, frc::Pose2d{8.5_m, 4_m, frc::Rotation2d{0_rad}});
  EXPECT_TRUE(tracker.GetClusters()->empty());
}
TEST(HSVClusterTrackerNodeTest, ProcessesPaddedDeviceFrameWithPoseAndPublishesOnce) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, {}, 640 * 480, 20, "pose");
  EXPECT_EQ(tracker.GetDependencies().size(), 2U);
  const auto image = SphereImage();
  auto frame = std::make_shared<camera::DecodedJpegBuffer>();
  frame->width = image.cols; frame->height = image.rows;
  frame->stride = image.cols * 3U + 16U;
  frame->output_format = NVJPEG_OUTPUT_BGRI;
  frame->output_size = frame->stride * frame->height;
  frame->channel_sizes[0] = frame->output_size;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&frame->destination.channel[0]), frame->output_size), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(frame->destination.channel[0], frame->stride, image.data,
      image.step, image.cols * 3U, image.rows, cudaMemcpyHostToDevice), cudaSuccess);
  auto context = std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 0);
  auto pose = std::make_shared<localization::PositionEstimateMessage>();
  pose->pose = frc::Pose3d{2_m, 4_m, 0_m, frc::Rotation3d{}};
  context->SetMessage("d", frame); context->SetMessage("pose", pose);
  int calls = 0;
  tracker.RegisterCallback([&](const auto&) { ++calls; });
  const auto callback = tracker.CreateCallback(); callback(context); callback(context);
  EXPECT_EQ(calls, 1);
  const auto* output = context->GetMessage<control_loop::ValueMessage<gamepiece::hsv_clusters_t>>("c");
  ASSERT_NE(output, nullptr); ASSERT_EQ(output->value.size(), 1U);
  EXPECT_NEAR(output->value.front().robot_relative_translation->X().value(), 4, 0.08);
}
TEST(HSVClusterTrackerTest, PreservesBroadPileAndFindsNewSeparatedPile) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, 0}, 0.9f, 0.25f);
  auto first = Cluster(tracker, points);
  ASSERT_EQ(first.size(), 1U);
  EXPECT_GT(first[0].ground_covar(0, 0), 0.6f);
  const auto new_points = Cloud({6, 3});
  points.insert(points.end(), new_points.begin(), new_points.end());
  auto next = Cluster(tracker, points);
  ASSERT_EQ(next.size(), 2U);
  EXPECT_EQ(next[0].id, first[0].id);
  EXPECT_NE(next[1].id, first[0].id);
  EXPECT_NEAR(next[1].robot_relative_translation->Y().value(), 3, 0.1);
}
TEST(HSVClusterTrackerTest, ProtectsWeakEnteringModeOutsideEstablishedFootprint) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, 0}, 0.4f, 0.35f);
  const auto first = Cluster(tracker, points);
  ASSERT_EQ(first.size(), 1U);
  auto entering = Cloud({6, 1.8f}, 0.5f, 0.25f);
  entering.resize(90);
  points.insert(points.end(), entering.begin(), entering.end());
  const auto next = Cluster(tracker, points);
  ASSERT_EQ(next.size(), 2U);
  EXPECT_EQ(next[0].id, first[0].id);
  EXPECT_NE(next[1].id, first[0].id);
  EXPECT_LT(next[0].ground_covar(1, 1), 0.2f);
}
TEST(HSVClusterTrackerTest, SeparatesSixPilesBeforeAnyIdsExist) {
  auto tracker = MakeTracker();
  std::vector<cv::Point2f> points;
  for (const auto center : {cv::Point2f{6, -1.5f}, {6, 0}, {6, 1.5f},
                            {9, -3}, {9, 0}, {9, 3}}) {
    const auto cloud = Cloud(center, 0.2f, 0.28f);
    points.insert(points.end(), cloud.begin(), cloud.end());
  }
  const auto first = Cluster(tracker, points);
  ASSERT_EQ(first.size(), 6U);
  const auto next = Cluster(tracker, points);
  ASSERT_EQ(next.size(), first.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(next[i].id, first[i].id);
    EXPECT_LT(next[i].ground_covar(1, 1), 0.15f);
  }
}
TEST(HSVClusterTrackerTest, UsesOneToOneIdsAndRetainsThemAcrossEmptyFrames) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, -1.5});
  const auto other = Cloud({6, 1.5});
  points.insert(points.end(), other.begin(), other.end());
  const auto first = Cluster(tracker, points);
  ASSERT_EQ(first.size(), 2U);
  tracker.ProcessFrame({});
  const auto next = Cluster(tracker, points);
  ASSERT_EQ(next.size(), 2U);
  EXPECT_EQ(first[0].id, next[0].id);
  EXPECT_EQ(first[1].id, next[1].id);
}
TEST(HSVClusterTrackerTest, MatchesFieldPositionsDuringRobotMotion) {
  auto tracker = MakeTracker();
  const auto points = Cloud({8, 4});
  const frc::Pose2d first_pose{1_m, 2_m, frc::Rotation2d{0_rad}};
  const auto first = Cluster(tracker, points, first_pose);
  ASSERT_EQ(first.size(), 1U);
  const auto next = Cluster(tracker, points, frc::Pose2d{3_m, 3_m, frc::Rotation2d{1_rad}});
  ASSERT_EQ(next.size(), 1U);
  EXPECT_EQ(first[0].id, next[0].id);
  const auto recovered = frc::Translation2d{3_m, 3_m} +
      next[0].robot_relative_translation->RotateBy(frc::Rotation2d{1_rad});
  EXPECT_NEAR(recovered.X().value(), 8, 0.1);
  EXPECT_NEAR(recovered.Y().value(), 4, 0.1);
}
TEST(HSVClusterTrackerTest, RejectsSingletonSpecksButRetainsPileSamples) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, 0});
  points.push_back({10, 4});
  auto result = Cluster(tracker, points);
  ASSERT_EQ(result.size(), 1U);
  EXPECT_NEAR(result[0].fuel_count, 30, 0.2);
  EXPECT_EQ(result[0].ground_weights.size(), result[0].ground_points.size());
  EXPECT_TRUE(cv::checkRange(result[0].ground_covar));
}
TEST(HSVClusterTrackerTest, ProjectsOntoFuelCenterPlaneInRobotAxes) {
  auto tracker = MakeTracker();
  const auto point = tracker.RobotRelativeOffset({0.2f, 0.5f});
  EXPECT_NEAR(point.X().value(), 1.85, 1e-6);
  EXPECT_NEAR(point.Y().value(), -0.37, 1e-6);
  EXPECT_THROW(Tracker("d", "c", kIntrinsics, kExtrinsics, {}, 0), std::invalid_argument);
  EXPECT_THROW(Tracker("d", "c", kIntrinsics, kExtrinsics, {}, 1, 0), std::invalid_argument);
}
TEST(HSVClusterTrackerTest, HandlesEmptyThresholdAndHorizonWithoutClusters) {
  auto tracker = MakeTracker();
  cv::Mat3b image(96, 128, cv::Vec3b{0, 0, 0});
  image(cv::Rect{20, 0, 80, 30}).setTo(cv::Scalar{0, 255, 255});
  tracker.ProcessFrame(image);
  EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame({});
  EXPECT_TRUE(tracker.GetClusters()->empty());
}
TEST(HSVClusterTrackerNodeTest, DeclaresMessagesAndPublishesNullOnce) {
  auto tracker = MakeTracker();
  EXPECT_EQ(tracker.GetDependencies().size(), 1U);
  EXPECT_EQ(tracker.GetPublications().size(), 1U);
  int calls = 0;
  tracker.RegisterCallback([&](const auto&) { ++calls; });
  auto context = std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 0);
  tracker.CreateCallback()(context);
  tracker.CreateCallback()(context);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(context->Exists("clusters"));
}
}
