#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <stop_token>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {
struct HSVClusterTrackerNodeTestPeer {
  static auto Run(const HSVClusterTrackerNode& tracker,
                  const std::vector<cv::Point2f>& points,
                  const hsv_clusters_t& seeds) {
    return tracker.KMeans(points, seeds);
  }
  static auto Track(HSVClusterTrackerNode& tracker, hsv_clusters_t clusters) {
    return tracker.AssignPersistentIds(std::move(clusters));
  }
};
}  // namespace gamepiece

namespace {

const cv::Matx33d kIntrinsics{100, 0, 64, 0, 100, 48, 0, 0, 1};
const cv::Matx44d kExtrinsics{1, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 1, 1, 0, 0, 0, 1};
const cv::Vec<double, 5> kDistortion{0, 0, 0, 0, 0};
constexpr std::size_t kImageSize = 128 * 96;

auto MakeContext(std::uint64_t id = 0) -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, id);
}

auto MakeImage() -> cv::Mat {
  cv::Mat3b hsv(96, 128, cv::Vec3b{0, 0, 0});
  hsv(cv::Rect{20, 60, 20, 20}).setTo(cv::Scalar{18, 255, 255});
  hsv(cv::Rect{80, 60, 20, 20}).setTo(cv::Scalar{30, 255, 255});
  hsv(cv::Rect{45, 60, 10, 20}).setTo(cv::Scalar{17, 255, 255});
  hsv(cv::Rect{60, 60, 10, 20}).setTo(cv::Scalar{31, 255, 255});
  hsv(cv::Rect{20, 30, 20, 20}).setTo(cv::Scalar{24, 140, 255});
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  return bgr;
}

auto GroundCluster(float lateral, float forward = 2.0f)
    -> gamepiece::kmeans_cluster_t {
  gamepiece::kmeans_cluster_t cluster;
  cluster.centroid = {-lateral / forward, 1.0f / forward};
  cluster.img_points = {cluster.centroid - cv::Point2f{0.002f, 0},
                        cluster.centroid + cv::Point2f{0.002f, 0}};
  return cluster;
}

TEST(HSVClusterTrackerTest, PersistentIdsFollowNearestProjectedCentroids) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  const auto track = [&](gamepiece::hsv_clusters_t clusters) {
    return gamepiece::HSVClusterTrackerNodeTestPeer::Track(tracker,
                                                          std::move(clusters));
  };
  const auto first = track({GroundCluster(0), GroundCluster(0.4f)});
  ASSERT_EQ(first.size(), 2U);  // New clusters do not match each other.
  EXPECT_NE(first[0].id, first[1].id);
  auto next = track({GroundCluster(0.35f), GroundCluster(0.05f)});
  ASSERT_EQ(next.size(), 2U);
  EXPECT_EQ(next[0].id, first[1].id);
  EXPECT_EQ(next[1].id, first[0].id);
  for (float lateral : {0.75f, 1.15f}) {
    next = track({GroundCluster(lateral)});
    ASSERT_EQ(next.size(), 1U);
    EXPECT_EQ(next[0].id, first[1].id);
  }
  tracker.ProcessFrame(cv::Mat{});
  tracker.ProcessFrame(cv::Mat3b::zeros(96, 128));
  next = track({GroundCluster(0.05f)});
  ASSERT_EQ(next.size(), 1U);
  EXPECT_EQ(next[0].id, first[0].id);
}

TEST(HSVClusterTrackerTest, IdToleranceUsesGroundMetersAndCreatesNewIds) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  const auto track = [&](float lateral, float forward) {
    return gamepiece::HSVClusterTrackerNodeTestPeer::Track(
        tracker, {GroundCluster(lateral, forward)}).front().id;
  };
  const auto original = track(0, 10);
  EXPECT_NE(track(0.6f, 10), original);  // Only 0.06 apart in image coordinates.
  EXPECT_EQ(track(-0.4f, 10), original);
  const auto fresh = track(0, 2);
  EXPECT_NE(fresh, original);
  EXPECT_EQ(track(0.5f, 2), fresh);  // Inclusive tolerance boundary.
}

TEST(HSVClusterTrackerTest, MergesSameIdClustersAndRecomputesGeometry) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  const auto first = gamepiece::HSVClusterTrackerNodeTestPeer::Track(
      tracker, {GroundCluster(0)});
  auto right = GroundCluster(0.2f);
  right.img_points.push_back(right.centroid);
  const auto merged = gamepiece::HSVClusterTrackerNodeTestPeer::Track(
      tracker, {GroundCluster(-0.4f), right});
  ASSERT_EQ(merged.size(), 1U);
  EXPECT_EQ(merged[0].id, first[0].id);
  EXPECT_EQ(merged[0].img_points.size(), 5U);
  EXPECT_NEAR(merged[0].centroid.x, 0.02f, 1e-6);
  EXPECT_NEAR(merged[0].covar(0, 0), 0.0216032f, 1e-6);
  ASSERT_TRUE(merged[0].robot_relative_translation.has_value());
  EXPECT_NEAR(merged[0].robot_relative_translation->Y().value(), -0.04, 1e-6);
}

TEST(HSVClusterTrackerTest, SplitsOnlySpikingClustersAlongTheirMajorAxis) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  const auto seed = [](float x) {
    gamepiece::kmeans_cluster_t cluster;
    cluster.centroid = {x, 0.5f};
    cluster.covar = cv::Matx22f::eye() * 0.0001f;
    return cluster;
  };
  std::vector<cv::Point2f> points;
  for (float x : {-0.6f, -0.2f, 0.4f}) {
    for (float dx : {-0.005f, 0.005f}) {
      for (float dy : {-0.005f, 0.005f}) {
        points.emplace_back(x + dx, 0.5f + dy);
      }
    }
  }
  auto clusters = gamepiece::HSVClusterTrackerNodeTestPeer::Run(
      tracker, points, {seed(-0.4f), seed(0.4f)});
  ASSERT_EQ(clusters.size(), 3U);
  std::ranges::sort(clusters, {}, [](const auto& c) { return c.centroid.x; });
  const float expected_x[] = {-0.6f, -0.2f, 0.4f};
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(clusters[i].img_points.size(), 4U);
    EXPECT_NEAR(clusters[i].centroid.x, expected_x[i], 1e-5);
    EXPECT_NEAR(clusters[i].centroid.y, 0.5f, 1e-5);
  }
  EXPECT_EQ(gamepiece::HSVClusterTrackerNodeTestPeer::Run(tracker, points,
                                                        clusters).size(), 3U);
  EXPECT_TRUE(gamepiece::HSVClusterTrackerNodeTestPeer::Run(
                  tracker, {points.front()}, clusters).empty());
}

TEST(HSVClusterTrackerTest, DropsDeadSeedsAndClampsAboveHorizonSeeds) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  const std::vector<cv::Point2f> points{{-0.01f, 0.5f}, {0.01f, 0.5f},
                                       {-0.01f, 0.6f}, {0.01f, 0.6f}};
  gamepiece::hsv_clusters_t seeds(6);
  for (auto& seed : seeds) {
    seed.centroid = {0, 0.55f};
  }
  auto clusters = gamepiece::HSVClusterTrackerNodeTestPeer::Run(
      tracker, points, seeds);
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front().img_points.size(), points.size());
  seeds.resize(1);
  for (float y : {-0.2f, 0.0f, 0.001f}) {
    seeds.front().centroid.y = y;
    clusters = gamepiece::HSVClusterTrackerNodeTestPeer::Run(
        tracker, points, seeds);
    ASSERT_EQ(clusters.size(), 1U);
    EXPECT_EQ(clusters.front().img_points.size(), points.size());
    EXPECT_TRUE(cv::checkRange(clusters.front().covar));
  }
  seeds.front().centroid = {0, 2.0f};
  seeds.front().covar = cv::Matx22f::eye() * 0.0001f;
  clusters = gamepiece::HSVClusterTrackerNodeTestPeer::Run(tracker, points, seeds);
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front().img_points.size(), points.size());
}

TEST(HSVClusterTrackerTest, CullsUsingFullImagePixelCountAfterKMeans) {
  const std::vector<cv::Point2f> points{{-0.5f, 0.5f}, {-0.51f, 0.5f},
                                       {0.5f, 0.5f}, {0.51f, 0.5f},
                                       {0.5f, 0.51f}};
  gamepiece::hsv_clusters_t seeds(2);
  seeds[0].centroid = {-0.5f, 0.5f};
  seeds[1].centroid = {0.5f, 0.5f};
  for (std::size_t area : {20000U, 20001U}) {
    gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                           kExtrinsics, kDistortion, area);
    const auto clusters = gamepiece::HSVClusterTrackerNodeTestPeer::Run(
        tracker, points, seeds);
    ASSERT_EQ(clusters.size(), area == 20000 ? 2U : 1U);
    EXPECT_EQ(clusters.back().img_points.size(), 3U);
  }
}

TEST(HSVClusterTrackerTest, PreservesBosThresholdsAndClearsEmptyFrames) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  tracker.ProcessFrame(MakeImage());
  ASSERT_FALSE(tracker.GetClusters()->empty());
  std::size_t points = 0;
  for (const auto& cluster : *tracker.GetClusters()) {
    points += cluster.img_points.size();
    ASSERT_TRUE(cluster.robot_relative_translation.has_value());
    EXPECT_TRUE(std::isfinite(cluster.robot_relative_translation->X().value()));
    EXPECT_TRUE(std::isfinite(cluster.robot_relative_translation->Y().value()));
    EXPECT_TRUE(cv::checkRange(cluster.covar));
    EXPECT_GT(cv::trace(cluster.covar), 0);
    for (const auto& point : cluster.img_points) {
      EXPECT_GE(point.y, 0.12f);
      EXPECT_LT(point.y, 0.32f);
      EXPECT_TRUE((point.x >= -0.44f && point.x < -0.24f) ||
                  (point.x >= 0.16f && point.x < 0.36f));
    }
  }
  EXPECT_EQ(points, 800U);
  tracker.ProcessFrame(cv::Mat{});
  EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame(cv::Mat3b::zeros(96, 128));
  EXPECT_TRUE(tracker.GetClusters()->empty());
  tracker.ProcessFrame(MakeImage());
  EXPECT_FALSE(tracker.GetClusters()->empty());
}

TEST(HSVClusterTrackerTest, StoresNormalizedCoordinatesAndProjectsCentroids) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  tracker.ProcessFrame(MakeImage());
  ASSERT_FALSE(tracker.GetClusters()->empty());
  for (const auto& cluster : *tracker.GetClusters()) {
    ASSERT_FALSE(cluster.img_points.empty());
    ASSERT_TRUE(cluster.robot_relative_translation.has_value());
    // This calibration has no rotation and a one-meter camera height. A ray
    // intersects the floor at forward distance height / normalized_y.
    const double forward_distance = 1.0 / cluster.centroid.y;
    const double lateral_distance = -cluster.centroid.x * forward_distance;
    EXPECT_NEAR(cluster.robot_relative_translation->X().value(),
                forward_distance, 1e-5);
    EXPECT_NEAR(cluster.robot_relative_translation->Y().value(),
                lateral_distance, 1e-5);
    EXPECT_GE(cluster.centroid.y, 0.12f);
    EXPECT_LT(cluster.centroid.y, 0.32f);
  }
}

TEST(HSVClusterTrackerTest, TranslationIsRobotRelativeInWpilibAxes) {
  // Camera 0.5 m forward and 0.2 m left of the robot origin, 1 m high.
  const cv::Matx44d offset_extrinsics{1, 0, 0, 0.5, 0, 1, 0, 0.2,
                                      0, 0, 1, 1,   0, 0, 0, 1};
  gamepiece::HSVClusterTrackerNode centered("decoded", "clusters", kIntrinsics,
                                            kExtrinsics, kDistortion,
                                            kImageSize);
  gamepiece::HSVClusterTrackerNode offset("decoded", "clusters", kIntrinsics,
                                          offset_extrinsics, kDistortion,
                                          kImageSize);
  centered.ProcessFrame(MakeImage());
  offset.ProcessFrame(MakeImage());
  ASSERT_EQ(centered.GetClusters()->size(), offset.GetClusters()->size());
  for (std::size_t i = 0; i < offset.GetClusters()->size(); ++i) {
    const auto& expected = *centered.GetClusters()->at(i).robot_relative_translation;
    const auto& actual = *offset.GetClusters()->at(i).robot_relative_translation;
    EXPECT_NEAR(actual.X().value(), expected.X().value() + 0.5, 1e-5);
    EXPECT_NEAR(actual.Y().value(), expected.Y().value() + 0.2, 1e-5);
  }
}

TEST(HSVClusterTrackerTest, NormalizedPointsReprojectWithDistortion) {
  const cv::Matx33d intrinsics{140, 0, 64, 0, 90, 48, 0, 0, 1};
  const cv::Vec<double, 5> distortion{0.1, -0.03, 0.01, -0.02, 0.005};
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", intrinsics,
                                         kExtrinsics, distortion, kImageSize);
  tracker.ProcessFrame(MakeImage());
  ASSERT_FALSE(tracker.GetClusters()->empty());
  std::size_t point_count = 0;
  for (const auto& cluster : *tracker.GetClusters()) {
    std::vector<cv::Point3f> rays;
    cv::Point2f sum{0, 0};
    for (const auto& point : cluster.img_points) {
      rays.emplace_back(point.x, point.y, 1.0f);
      sum += point;
    }
    const cv::Point2f mean =
        sum / static_cast<float>(cluster.img_points.size());
    EXPECT_NEAR(cv::norm(mean - cluster.centroid), 0, 1e-6);
    std::vector<cv::Point2f> pixels;
    cv::projectPoints(rays, cv::Vec3d{}, cv::Vec3d{}, intrinsics, distortion,
                      pixels);
    for (const auto& pixel : pixels) {
      EXPECT_NEAR(pixel.x, std::round(pixel.x), 1e-3);
      EXPECT_NEAR(pixel.y, std::round(pixel.y), 1e-3);
      EXPECT_GE(std::round(pixel.y), 60);
      EXPECT_LT(std::round(pixel.y), 80);
      EXPECT_TRUE((std::round(pixel.x) >= 20 && std::round(pixel.x) < 40) ||
                  (std::round(pixel.x) >= 80 && std::round(pixel.x) < 100));
    }
    point_count += pixels.size();
  }
  EXPECT_EQ(point_count, 800U);
}

TEST(HSVClusterTrackerTest, RejectsHorizonPointsAndCapsClustersToUsablePoints) {
  cv::Mat3b hsv(96, 128, cv::Vec3b{0, 0, 0});
  hsv(30, 64) = cv::Vec3b{24, 255, 255};
  hsv(48, 64) = cv::Vec3b{24, 255, 255};
  hsv(49, 64) = cv::Vec3b{24, 255, 255};  // projects 100 meters away
  hsv(70, 64) = cv::Vec3b{24, 255, 255};
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, 1);
  tracker.ProcessFrame(MakeImage());
  tracker.ProcessFrame(bgr);
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  const auto& cluster = tracker.GetClusters()->front();
  ASSERT_EQ(cluster.img_points.size(), 1U);
  EXPECT_EQ(cluster.img_points.front(), cv::Point2f(0, 0.22f));
  EXPECT_EQ(cluster.centroid, cv::Point2f(0, 0.22f));
  EXPECT_TRUE(cv::checkRange(cluster.covar));
  EXPECT_EQ(cv::norm(cluster.covar), 0);
  hsv(70, 64) = cv::Vec3b{0, 0, 0};
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  tracker.ProcessFrame(bgr);
  EXPECT_TRUE(tracker.GetClusters()->empty());
}

TEST(HSVClusterTrackerTest, AppliesConfigurableGroundDistanceLimit) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize,
                                         2.0f);
  tracker.ProcessFrame(MakeImage());
  EXPECT_TRUE(tracker.GetClusters()->empty());
  EXPECT_THROW(gamepiece::HSVClusterTrackerNode("decoded", "clusters",
                                               kIntrinsics, kExtrinsics,
                                               kDistortion, kImageSize, 0.0f),
               std::invalid_argument);
}

TEST(HSVClusterTrackerTest, ColumnBottomsKeepValidPointsAcrossTheHorizon) {
  for (bool rolled : {false, true}) {
    cv::Mat3b hsv(96, 128, cv::Vec3b{0, 0, 0});
    auto extrinsics = kExtrinsics;
    if (rolled) {
      extrinsics = {1, 0, 0, 0, 0, 0.6, 0.8, 0,
                    0, -0.8, 0.6, 0.1, 0, 0, 0, 1};
      // All rays are valid, but extrapolating column 74 crosses the horizon.
      hsv(63, 74) = cv::Vec3b{24, 255, 255};
      hsv(cv::Rect{75, 78, 3, 1}).setTo(cv::Scalar{24, 255, 255});
    } else {
      // Only the noise column was sampled before filtering invalid rays.
      hsv(30, 0) = cv::Vec3b{24, 255, 255};
      hsv(cv::Rect{1, 70, 3, 1}).setTo(cv::Scalar{24, 255, 255});
    }
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                           extrinsics, kDistortion, 1);
    ASSERT_NO_THROW(tracker.ProcessFrame(bgr));
    std::size_t count = 0;
    for (const auto& cluster : *tracker.GetClusters()) {
      count += cluster.img_points.size();
      ASSERT_TRUE(cluster.robot_relative_translation.has_value());
      EXPECT_TRUE(std::isfinite(cluster.robot_relative_translation->Norm().value()));
    }
    EXPECT_EQ(count, rolled ? 4U : 3U);
  }
}

TEST(HSVClusterTrackerTest, ClusteringIsIndependentOfPixelOrigin) {
  const cv::Mat image = MakeImage();
  cv::Mat3b shifted_image(144, 192, cv::Vec3b{0, 0, 0});
  image.copyTo(shifted_image(cv::Rect{64, 48, image.cols, image.rows}));
  const cv::Matx33d intrinsics{128, 0, 64, 0, 128, 48, 0, 0, 1};
  const cv::Matx33d shifted_intrinsics{128, 0, 128, 0, 128, 96, 0, 0, 1};
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", intrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  gamepiece::HSVClusterTrackerNode shifted_tracker(
      "decoded", "clusters", shifted_intrinsics, kExtrinsics, kDistortion,
      shifted_image.total());
  tracker.ProcessFrame(image);
  shifted_tracker.ProcessFrame(shifted_image);
  ASSERT_FALSE(tracker.GetClusters()->empty());
  ASSERT_EQ(tracker.GetClusters()->size(), shifted_tracker.GetClusters()->size());
  for (std::size_t i = 0; i < tracker.GetClusters()->size(); ++i) {
    const auto& original = tracker.GetClusters()->at(i);
    const auto& shifted = shifted_tracker.GetClusters()->at(i);
    ASSERT_EQ(original.img_points.size(), shifted.img_points.size());
    for (std::size_t point = 0; point < original.img_points.size(); ++point) {
      EXPECT_EQ(original.img_points[point], shifted.img_points[point]);
    }
    EXPECT_NEAR(cv::norm(original.centroid - shifted.centroid), 0, 1e-3);
  }
}

TEST(HSVClusterTrackerNodeTest, DeclaresInputAndOutputAndPublishesNullOnce) {
  gamepiece::HSVClusterTrackerNode node("decoded", "clusters", kIntrinsics,
                                        kExtrinsics, kDistortion, kImageSize);
  ASSERT_EQ(node.GetDependencies().size(), 1U);
  EXPECT_TRUE(node.GetDependencies()[0].GetTypes().contains(
      typeid(camera::DecodedJpegBuffer)));
  ASSERT_EQ(node.GetPublications().size(), 1U);
  EXPECT_TRUE(node.GetPublications()[0].GetTypes().contains(
      typeid(control_loop::ValueMessage<gamepiece::hsv_clusters_t>)));
  int notifications = 0;
  node.RegisterCallback([&](const control_loop::Context& context) -> void {
    EXPECT_TRUE(context->Exists("clusters"));
    EXPECT_EQ(context->GetMessage<
                  control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(
                  "clusters"), nullptr);
    ++notifications;
  });
  const auto run = node.CreateCallback();
  auto missing = MakeContext();
  run(missing);
  run(missing);
  auto null_frame = MakeContext();
  null_frame->SetMessage("decoded", nullptr);
  run(null_frame);
  auto invalid_frame = MakeContext();
  invalid_frame->SetMessage("decoded",
                           std::make_shared<camera::DecodedJpegBuffer>());
  run(invalid_frame);
  EXPECT_EQ(notifications, 3);
}

TEST(HSVClusterTrackerNodeTest, ProcessesPaddedDeviceFramesAndRetainsHistory) {
  gamepiece::HSVClusterTrackerNode node("decoded", "clusters", kIntrinsics,
                                        kExtrinsics, kDistortion, kImageSize);
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, kImageSize);
  int notifications = 0;
  node.RegisterCallback(
      [&](const control_loop::Context&) -> void { ++notifications; });
  const auto run = node.CreateCallback();
  std::shared_ptr<control_loop::ValueMessage<gamepiece::hsv_clusters_t>> first;
  for (std::uint64_t id = 0; id < 4; ++id) {
    cv::Mat image = MakeImage();
    if (id == 1) {
      image(cv::Rect{0, 0, 64, 96}).setTo(cv::Scalar{});
    } else if (id == 2) {
      image.setTo(cv::Scalar{});
    }
    auto frame = std::make_shared<camera::DecodedJpegBuffer>();
    frame->width = image.cols;
    frame->height = image.rows;
    frame->stride = static_cast<std::size_t>(image.cols) * 3U + 16U;
    frame->output_format = NVJPEG_OUTPUT_BGRI;
    frame->output_size = frame->stride * frame->height;
    frame->channel_sizes[0] = frame->output_size;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&frame->destination.channel[0]),
                         frame->output_size), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(frame->destination.channel[0], frame->stride,
                           image.data, image.step,
                           static_cast<std::size_t>(image.cols) * 3U,
                           image.rows, cudaMemcpyHostToDevice), cudaSuccess);
    const auto context = MakeContext(id);
    context->SetMessage("decoded", frame);
    run(context);
    run(context);
    tracker.ProcessFrame(image);
    const auto output = context->GetSharedMessage<
        control_loop::ValueMessage<gamepiece::hsv_clusters_t>>("clusters");
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->value.size(), tracker.GetClusters()->size());
    for (std::size_t i = 0; i < output->value.size(); ++i) {
      const auto& actual = output->value[i];
      const auto& expected = tracker.GetClusters()->at(i);
      EXPECT_EQ(actual.id, expected.id);
      EXPECT_EQ(actual.centroid, expected.centroid);
      EXPECT_EQ(actual.img_points, expected.img_points);
      EXPECT_DOUBLE_EQ(cv::norm(actual.covar - expected.covar), 0);
      EXPECT_EQ(actual.robot_relative_translation,
                expected.robot_relative_translation);
    }
    if (id == 0) {
      first = output;
    }
  }
  EXPECT_EQ(notifications, 4);
  ASSERT_NE(first, nullptr);
  std::size_t first_points = 0;
  for (const auto& cluster : first->value) {
    first_points += cluster.img_points.size();
    EXPECT_TRUE(cv::checkRange(cluster.covar));
  }
  EXPECT_EQ(first_points, 800U);
}

}  // namespace
