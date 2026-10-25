#include "gamepiece/hsv_cluster_tracker_node.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <numbers>
#include <random>
#include <frc/geometry/struct/Pose2dStruct.h>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include "localization/position.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "logging/wpilog_writer.h"
#include "streamer/densest_cluster_rio_streamer_node.h"

namespace gamepiece {
struct HSVClusterTrackerNodeTestPeer {
  static auto Pose(const HSVClusterTrackerNode& tracker) { return tracker.last_robot_pose_; }
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
using Clock = std::chrono::steady_clock;
const cv::Matx33d kIntrinsics{100, 0, 64, 0, 100, 48, 0, 0, 1};
const cv::Matx44d kExtrinsics{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
using Tracker = gamepiece::HSVClusterTrackerNode;
auto MakeTracker() { return Tracker("decoded", "clusters", kIntrinsics, kExtrinsics, 128 * 96); }
TEST(HSVClusterTrackerNodeTest, ProductionPipelineLoggingStarts) {
  control_loop::ControlLoop loop(std::chrono::milliseconds{1});
  auto tracker = std::make_shared<Tracker>("decoded", "clusters", kIntrinsics, kExtrinsics, 128 * 96);
  tracker->EnablePoseOutput("frame/pose"); loop.RegisterNode(tracker);
  loop.RegisterNode(std::make_shared<streamer::DensestClusterRioStreamerNode>("clusters", "frame/pose", "/COS/DensestCluster"));
  const auto path = std::filesystem::temp_directory_path() / "cos-cluster-startup.wpilog";
  const auto publications = loop.GetLogPublications();
  ASSERT_EQ(publications.size(), 2U);
  EXPECT_NO_THROW({ logging::WPILogWriter writer(path.string(), publications); });
  std::filesystem::remove(path);
}
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
TEST(HSVClusterTrackerTest, SeparatesCompactPilesAcrossShallowSmoothedValley) {
  auto tracker = MakeTracker();
  auto points = Cloud({6, -0.65f}, 0.2f, 0.25f);
  const auto other = Cloud({6, 0.65f}, 0.2f, 0.25f);
  points.insert(points.end(), other.begin(), other.end());
  EXPECT_EQ(Cluster(tracker, points).size(), 2U);
}
TEST(HSVClusterTrackerTest, RejectsEmptyHorizonAndSingletonEvidence) {
  EXPECT_THROW(Tracker("d", "c", kIntrinsics, kExtrinsics, 0), std::invalid_argument);
  auto tracker = MakeTracker();
  cv::Mat3b image(96, 128, cv::Vec3b{});
  image(cv::Rect{20, 0, 80, 30}).setTo(cv::Scalar{0, 255, 255});
  tracker.ProcessFrame(image); EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame({}); EXPECT_TRUE(tracker.GetClusters()->empty());
  Tracker spheres("d", "c", kSphereIntrinsics, kExtrinsics, 1920 * 1080);
  spheres.ProcessFrame(SphereImage(6, 1)); EXPECT_TRUE(spheres.GetClusters()->empty());
  spheres.ProcessFrame(SphereImage(6, 2)); ASSERT_EQ(spheres.GetClusters()->size(), 1U);
  EXPECT_NEAR(spheres.GetClusters()->front().robot_relative_translation->X().value(), 6, .25);
}
TEST(HSVClusterTrackerTest, RetainsOutOfViewHistoryButResetsCoordinateModesAndExpiredEvidence) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  const frc::Pose2d pose{2_m, 4_m, frc::Rotation2d{0_rad}};
  const auto image = SphereImage(); const cv::Mat3b empty(480, 640, cv::Vec3b{});
  tracker.ProcessFrame(image, pose);
  ASSERT_EQ(tracker.GetClusters()->size(), 1U); const auto id = tracker.GetClusters()->front().id;
  for (int i = 0; i < 20; ++i) tracker.ProcessFrame(empty, {frc::Pose2d{2_m, 4_m, frc::Rotation2d{3.14159_rad}}});
  tracker.ProcessFrame(image, pose); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().id, id); EXPECT_GT(tracker.GetClusters()->front().reused_ball_count, 0U);
  tracker.ProcessFrame(image); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().reused_ball_count, 0U);
  for (int i = 0; i < 9; ++i) tracker.ProcessFrame(empty);
  tracker.ProcessFrame(image); ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  EXPECT_EQ(tracker.GetClusters()->front().reused_ball_count, 0U);
}
TEST(HSVClusterTrackerTest, DensityHistoryFollowsRemovalAndAddition) {
  Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480);
  const auto settle = [&](int side, int frames) {
    const auto image = GridImage(side);
    for (int i = 0; i < frames; ++i) tracker.ProcessFrame(image);
    EXPECT_EQ(tracker.GetClusters()->size(), 1U);
    return tracker.GetClusters()->empty() ? 0.f : tracker.GetClusters()->front().estimated_ball_count;
  };
  const float before = settle(5, 8), after = settle(3, 6);
  EXPECT_LT(after, .65f * before); EXPECT_NEAR(after, 9, 4.5);
  EXPECT_GT(settle(5, 6), .85f * before);
}
TEST(HSVClusterTrackerNodeTest, PullsLatestNetworkTablesPoseOnlyWhenProcessingAFrame) {
  auto instance = nt::NetworkTableInstance::Create();
  {
    Tracker tracker("d", "c", kIntrinsics, kExtrinsics, 128 * 96, "",
                    gamepiece::DensityHistory::kPredict, true, "/test/Pose", instance);
    tracker.EnablePoseOutput("frame/pose");
    EXPECT_EQ(tracker.GetPublications().size(), 2U);
    auto publisher = instance.GetStructTopic<frc::Pose2d>("/test/Pose").Publish();
    const auto callback = tracker.CreateCallback();
    const auto frame = [&] {
      auto context = std::make_shared<control_loop::ContextInternal>(Clock::now(), nullptr, std::stop_token{}, 0);
      context->SetMessage("d", nullptr); callback(context); callback(context);
      EXPECT_TRUE(context->Exists("frame/pose"));
      const auto* output = context->GetMessage<localization::PositionEstimateMessage>("frame/pose");
      const auto pose = gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker);
      if (pose) { ASSERT_NE(output, nullptr); EXPECT_EQ(output->pose.ToPose2d(), *pose); }
      else EXPECT_EQ(output, nullptr);
    };
    EXPECT_EQ(tracker.GetDependencies().size(), 1U);
    frame(); EXPECT_FALSE(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker));
    for (double x : {1.0, 2.0}) publisher.Set(frc::Pose2d{units::meter_t{x}, 3_m, frc::Rotation2d{}});
    frame(); ASSERT_TRUE(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker));
    EXPECT_DOUBLE_EQ(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker)->X().value(), 2);
    publisher.Set(frc::Pose2d{4_m, 5_m, frc::Rotation2d{0.7_rad}});
    EXPECT_DOUBLE_EQ(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker)->X().value(), 2);
    frame(); EXPECT_DOUBLE_EQ(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker)->X().value(), 4);
    EXPECT_NEAR(gamepiece::HSVClusterTrackerNodeTestPeer::Pose(tracker)->Rotation().Radians().value(), .7, 1e-9);
  }
  nt::NetworkTableInstance::Destroy(instance);
}
TEST(HSVClusterTrackerNodeTest, WaitsForBothDependenciesAndPublishesPaddedFramesOnce) {
  const auto image = SphereImage();
  auto frame = std::make_shared<camera::DecodedJpegBuffer>();
  frame->width = image.cols; frame->height = image.rows; frame->stride = image.cols * 3U + 16U;
  frame->output_format = NVJPEG_OUTPUT_BGRI; frame->channel_sizes[0] = frame->output_size = frame->stride * frame->height;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&frame->destination.channel[0]), frame->output_size), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(frame->destination.channel[0], frame->stride, image.data, image.step,
                        image.cols * 3U, image.rows, cudaMemcpyHostToDevice), cudaSuccess);
  for (bool pose_first : {false, true}) {
    Tracker tracker("d", "c", kSphereIntrinsics, kExtrinsics, 640 * 480, "pose");
    tracker.EnablePoseOutput("frame/pose");
    tracker.EnableMaskOutput("mask");
    streamer::DensestClusterRioStreamerNode sender("c", "frame/pose", "/test/DensestCluster");
    auto subscriber = nt::NetworkTableInstance::GetDefault().GetDoubleArrayTopic("/test/DensestCluster")
        .Subscribe({}, {.sendAll = true, .keepDuplicates = true});
    tracker.RegisterCallback(sender.CreateCallback());
    auto context = std::make_shared<control_loop::ContextInternal>(Clock::now(), nullptr, std::stop_token{}, 0);
    auto pose = std::make_shared<localization::PositionEstimateMessage>();
    pose->pose = frc::Pose3d{2_m, 4_m, 0_m, frc::Rotation3d{}};
    int calls = 0; tracker.RegisterCallback([&](const auto&) { ++calls; });
    const auto callback = tracker.CreateCallback();
    if (pose_first) context->SetMessage("pose", pose); else context->SetMessage("d", frame);
    callback(context); EXPECT_FALSE(context->Exists("c")); EXPECT_EQ(calls, 0);
    if (pose_first) context->SetMessage("d", frame); else context->SetMessage("pose", pose);
    callback(context); callback(context); EXPECT_EQ(calls, 1);
    const auto* output = context->GetMessage<control_loop::ValueMessage<gamepiece::hsv_clusters_t>>("c");
    ASSERT_NE(output, nullptr); ASSERT_EQ(output->value.size(), 1U);
    EXPECT_GT(output->value.front().density, 0);
    const auto* mask = context->GetMessage<control_loop::ValueMessage<cv::Mat>>("mask");
    ASSERT_NE(mask, nullptr); EXPECT_EQ(mask->value.size(), image.size());
    EXPECT_EQ(mask->value.type(), CV_8UC1); EXPECT_GT(cv::countNonZero(mask->value), 0);
    tracker.ProcessFrame(cv::Mat{}); EXPECT_GT(cv::countNonZero(mask->value), 0);
    const auto samples = subscriber.ReadQueue();
    ASSERT_EQ(samples.size(), 1U); ASSERT_EQ(samples[0].value.size(), 10U);
    for (double value : samples[0].value) EXPECT_TRUE(std::isfinite(value));
    const auto center = *output->value.front().robot_relative_translation + pose->pose.ToPose2d().Translation();
    EXPECT_NEAR(samples[0].value[0], center.X().value(), 1e-6);
    EXPECT_NEAR(samples[0].value[1], center.Y().value(), 1e-6);
  }
  auto tracker = MakeTracker(); int calls = 0; tracker.RegisterCallback([&](const auto&) { ++calls; });
  auto context = std::make_shared<control_loop::ContextInternal>(Clock::now(), nullptr, std::stop_token{}, 0);
  context->SetMessage("decoded", nullptr);
  const auto callback = tracker.CreateCallback(); callback(context); callback(context);
  EXPECT_TRUE(context->Exists("clusters")); EXPECT_EQ(calls, 1);
}
}
