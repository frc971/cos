#include "gamepiece/hsv_cluster_tracker_node.h"

#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include "localization/position.h"
#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {
struct HSVClusterTrackerNodeTestPeer {
  static auto ArcWeight(const cv::Matx33f& K, const cv::Vec3f& ray, const cv::Vec2f& normal, float depth) {
    return HSVClusterTrackerNode::SphereArcWeight(K, ray, normal, depth);
  }
  static auto Run(HSVClusterTrackerNode& tracker, const std::vector<cv::Point2f>& points,
                  std::optional<frc::Pose2d> pose = std::nullopt) {
    std::vector<cv::Point2f> image_points;
    const std::vector<float> weights(points.size(), 0.1f);
    for (const auto& p : points) image_points.push_back({-p.y / p.x, 0.925f / p.x});
    tracker.clusters_ = tracker.ClusterGroundPoints(image_points, points, weights, weights, pose);
    tracker.AssignPersistentIds(tracker.clusters_, pose);
    return *tracker.GetClusters();
  }
};
}
namespace {
const cv::Matx33d kIntrinsics{100, 0, 64, 0, 100, 48, 0, 0, 1};
const cv::Matx44d kExtrinsics{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
using Tracker = gamepiece::HSVClusterTrackerNode;
auto MakeTracker() { return Tracker("decoded", "clusters", kIntrinsics, kExtrinsics, 128 * 96); }
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
TEST(HSVClusterTrackerTest, ArcEvidenceConservesSphereMassOffAxis) {
  const cv::Matx33f K{600, 0, 320, 0, 430, 100, 0, 0, 1};
  for (const float angle : {0.0f, .35f, .61f, -.61f}) {
    const cv::Vec3f center{4*std::tan(angle), 0, 4}, axis = center / cv::norm(center);
    auto first = axis.cross(cv::Vec3f{0, 1, 0}); first /= cv::norm(first);
    const auto second = axis.cross(first); const float sine = .075f / cv::norm(center);
    const auto sample = [&](int i) {
      const float t = i*2*std::numbers::pi_v<float>/4096;
      auto ray = std::sqrt(1-sine*sine)*axis+sine*(std::cos(t)*first+std::sin(t)*second);
      return ray/ray[2];
    };
    float mass = 0;
    for (int i = 0; i < 4096; ++i) {
      const auto a = sample(i), b = sample(i+1), ray = (a+b)/2;
      const cv::Vec2f tangent{K(0, 0)*(b[0]-a[0]), K(1, 1)*(b[1]-a[1])};
      const float length = cv::norm(tangent);
      mass += length*gamepiece::HSVClusterTrackerNodeTestPeer::ArcWeight(K, ray, {-tangent[1]/length, tangent[0]/length}, 4);
    }
    EXPECT_NEAR(mass, 1, .003);
  }
}
auto Spheres(const std::vector<cv::Vec3f>& centers) -> cv::Mat {
  cv::Mat3b hsv(480, 640, cv::Vec3b{0, 0, 0});
  for (int y = 0; y < hsv.rows; ++y) for (int x = 0; x < hsv.cols; ++x) {
    const cv::Vec3f ray{(x - 320) / 400.0f, (y - 100) / 400.0f, 1};
    float nearest = 1e9f;
    for (const auto& center : centers) {
      const float dot = ray.dot(center);
      const float discriminant = dot * dot - ray.dot(ray) * (center.dot(center) - .075f*.075f);
      if (dot <= 0 || discriminant < 0) continue;
      const float t = (dot - std::sqrt(discriminant)) / ray.dot(ray);
      if (t >= nearest) continue;
      nearest = t;
      const auto normal = (t * ray - center) / .075f;
      hsv(y, x) = {24, 125, static_cast<uchar>(140 + 90*std::max(0.0f, normal.dot(cv::Vec3f{-.3f, -.4f, -.8660254f})))};
    }
  }
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  return bgr;
}
auto SphereImage(float forward = 4.0f, int balls = 6) -> cv::Mat {
  std::vector<cv::Vec3f> centers;
  for (int i = 0; i < balls; ++i) centers.push_back({0.45f - .18f*i, .925f, forward});
  return Spheres(centers);
}
auto GridImage(int side, float spread = .3f, float forward = 4) -> cv::Mat {
  std::vector<cv::Vec3f> centers;
  for (int x = 0; x < side; ++x) for (int y = 0; y < side; ++y)
    centers.push_back({spread*(2*y/(side-1.0f)-1), .925f, forward+spread*(2*x/(side-1.0f)-1)});
  return Spheres(centers);
}
TEST(HSVClusterTrackerTest, DensityRespondsToCountAreaAndPerspective) {
  const auto measure = [](int side, float spread, float forward) {
    Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
    tracker.ProcessFrame(GridImage(side, spread, forward));
    EXPECT_EQ(tracker.GetClusters()->size(), 1U);
    if (!tracker.GetClusters()->empty()) {
      const float count = tracker.GetClusters()->front().estimated_ball_count;
      EXPECT_NEAR(count, side*side, .5f*side*side);
      ::testing::Test::RecordProperty("balls_" + std::to_string(side) + "_" + std::to_string(spread) + "_" + std::to_string(forward), std::to_string(count));
    }
    return tracker.GetClusters()->empty() ? 0.0f : tracker.GetClusters()->front().density;
  };
  const float sparse = measure(3, .3f, 4), dense = measure(5, .3f, 4);
  const float broad = measure(3, .6f, 4), far = measure(3, .3f, 5);
  EXPECT_GT(dense / sparse, 2); EXPECT_LT(dense / sparse, 5);
  EXPECT_GT(sparse / broad, 2.5); EXPECT_LT(sparse / broad, 5.5);
  EXPECT_NEAR(far / sparse, 1, .3);
  RecordProperty("count_ratio", std::to_string(dense / sparse));
  RecordProperty("area_ratio", std::to_string(sparse / broad));
  RecordProperty("distance_ratio", std::to_string(far / sparse));
}
TEST(HSVClusterTrackerTest, SphereProjectionPreservesLocationAndRejectsAllianceZone) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
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
TEST(HSVClusterTrackerTest, PredictsStationaryBallsBeforeExploringNewEvidence) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  for (int i = 0; i < 6; ++i) {
    const float travel = .04f*i;
    tracker.ProcessFrame(SphereImage(4-travel), frc::Pose2d{units::meter_t{2+travel}, 4_m, frc::Rotation2d{0_rad}});
    ASSERT_EQ(tracker.GetClusters()->size(), 1U);
    if (i > 0) { EXPECT_GT(tracker.GetClusters()->front().reused_ball_count, 0U); }
  }
  const auto before = tracker.GetClusters()->front();
  tracker.ProcessFrame(cv::Mat3b(480, 640, cv::Vec3b{}), frc::Pose2d{2.2_m, 4_m, frc::Rotation2d{0_rad}});
  EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame(SphereImage(3.8f), frc::Pose2d{2.2_m, 4_m, frc::Rotation2d{0_rad}});
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().id, before.id);
  EXPECT_GT(tracker.GetClusters()->front().reused_ball_count, 0U);
  tracker.ProcessFrame(SphereImage(3.8f));  // Coordinate-mode changes invalidate history.
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().reused_ball_count, 0U);
  for (int i = 0; i < 9; ++i) tracker.ProcessFrame(cv::Mat3b(480, 640, cv::Vec3b{}));
  tracker.ProcessFrame(SphereImage(3.8f));
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().reused_ball_count, 0U);
}
TEST(HSVClusterTrackerTest, RetainsStationaryMapWhileCameraLooksAway) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  const frc::Pose2d pose{2_m, 4_m, frc::Rotation2d{0_rad}};
  tracker.ProcessFrame(SphereImage(), pose);
  for (int i = 0; i < 20; ++i) {
    tracker.ProcessFrame(cv::Mat3b(480, 640, cv::Vec3b{}), frc::Pose2d{2_m, 4_m, frc::Rotation2d{3.14159_rad}});
    EXPECT_TRUE(tracker.GetClusters()->empty());
  }
  tracker.ProcessFrame(SphereImage(), pose);
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_GT(tracker.GetClusters()->front().reused_ball_count, 0U);
}
TEST(HSVClusterTrackerTest, DensityHistoryRespondsToRemovalAndAddition) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  const auto dense = GridImage(5), sparse = GridImage(3);
  std::vector<float> counts;
  for (int i = 0; i < 8; ++i) {
    tracker.ProcessFrame(dense); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
    counts.push_back(tracker.GetClusters()->front().estimated_ball_count);
  }
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  const float before = tracker.GetClusters()->front().estimated_ball_count;
  for (int i = 0; i < 6; ++i) {
    tracker.ProcessFrame(sparse); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
    counts.push_back(tracker.GetClusters()->front().estimated_ball_count);
  }
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  const float after = tracker.GetClusters()->front().estimated_ball_count;
  EXPECT_LT(after, .65f*before); EXPECT_NEAR(after, 9, 4.5);
  for (int i = 0; i < 6; ++i) {
    tracker.ProcessFrame(dense); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
    counts.push_back(tracker.GetClusters()->front().estimated_ball_count);
  }
  EXPECT_GT(counts.back(), .85f*before);
  RecordProperty("count_before", std::to_string(before)); RecordProperty("count_after_6_frames", std::to_string(after));
  RecordProperty("count_returned_after_6_frames", std::to_string(counts.back()));
  for (std::size_t i = 0; i < counts.size(); ++i) RecordProperty("step_"+std::to_string(i), std::to_string(counts[i]));
}
TEST(HSVClusterTrackerTest, StationaryFootprintSurvivesChangingCameraYaw) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  float smallest = 1e9f, largest = 0;
  for (float angle : {0.0f, -.25f, -.15f, 0.0f, .15f, .25f}) {
    std::vector<cv::Vec3f> centers;
    for (int i = -1; i <= 1; ++i) for (int j = -3; j <= 3; ++j) {
      const float x = 4+.15f*i, y = .2f*j;
      centers.push_back({x*std::sin(angle)-y*std::cos(angle), .925f, x*std::cos(angle)+y*std::sin(angle)});
    }
    tracker.ProcessFrame(Spheres(centers), frc::Pose2d{2_m, 4_m, frc::Rotation2d{units::radian_t{angle}}});
    ASSERT_EQ(tracker.GetClusters()->size(), 1U);
    const auto& c = tracker.GetClusters()->front();
    smallest = std::min(smallest, c.density_area_m2); largest = std::max(largest, c.density_area_m2);
    if (angle != 0) { EXPECT_GT(c.reused_ball_count, 0U); }
  }
  EXPECT_LT(largest/smallest, 1.3f); RecordProperty("area_span", std::to_string(largest/smallest));
}
TEST(HSVClusterTrackerTest, DoesNotRejectSmallDistantPileBecauseOfImageArea) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 1920 * 1080);
  cv::Mat3b image(1080, 1920, cv::Vec3b{});
  SphereImage(6).copyTo(image(cv::Rect{0, 0, 640, 480}));
  tracker.ProcessFrame(image);
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_LT(tracker.GetClusters()->front().img_points.size(), 207U);
  EXPECT_NEAR(tracker.GetClusters()->front().robot_relative_translation->X().value(), 6, 0.25);
}
TEST(HSVClusterTrackerTest, RejectsOneBallButRetainsASmallPile) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  tracker.ProcessFrame(SphereImage(6, 1));
  EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame(SphereImage(6, 2));
  EXPECT_EQ(tracker.GetClusters()->size(), 1U);
}
TEST(HSVClusterTrackerNodeTest, ProcessesPaddedDeviceFrameWithPoseAndPublishesOnce) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480, "pose");
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
  EXPECT_GT(output->value.front().density, 0);
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
TEST(HSVClusterTrackerTest, SeparatesCompactPilesAcrossShallowSmoothedValley) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, -0.65f}, 0.2f, 0.25f);
  const auto other = Cloud({6, 0.65f}, 0.2f, 0.25f);
  points.insert(points.end(), other.begin(), other.end());
  EXPECT_EQ(Cluster(tracker, points).size(), 2U);
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
  EXPECT_THROW(Tracker("d", "c", kIntrinsics, kExtrinsics, 0), std::invalid_argument);
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
