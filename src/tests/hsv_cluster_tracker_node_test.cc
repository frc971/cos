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

namespace {

const cv::Matx33d kIntrinsics{100, 0, 64, 0, 100, 48, 0, 0, 1};
const cv::Matx44d kExtrinsics{1, 0, 0, 0, 0, 1, 0, 0,
                             0, 0, 1, 1, 0, 0, 0, 1};
const cv::Vec<double, 5> kDistortion{0, 0, 0, 0, 0};

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

TEST(HSVClusterTrackerTest, PreservesBosThresholdsAndClearsEmptyFrames) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion);
  tracker.ProcessFrame(MakeImage());
  ASSERT_FALSE(tracker.GetClusters()->empty());
  std::size_t points = 0;
  for (const auto& cluster : *tracker.GetClusters()) {
    points += cluster.img_points.size();
    ASSERT_TRUE(cluster.camera_relative_translation.has_value());
    EXPECT_TRUE(std::isfinite(cluster.camera_relative_translation->X().value()));
    EXPECT_TRUE(std::isfinite(cluster.camera_relative_translation->Y().value()));
    EXPECT_TRUE(cv::checkRange(cluster.covar));
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

TEST(HSVClusterTrackerTest, StoresNormalizedCoordinatesAndProjectsGroundRays) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion);
  tracker.ProcessFrame(MakeImage());
  ASSERT_FALSE(tracker.GetClusters()->empty());
  for (const auto& cluster : *tracker.GetClusters()) {
    ASSERT_FALSE(cluster.img_points.empty());
    ASSERT_TRUE(cluster.camera_relative_translation.has_value());
    const auto lowest_point = std::ranges::max_element(
        cluster.img_points, {}, &cv::Point2f::y);
    // This calibration has no rotation and a one-meter camera height. A ray
    // intersects the floor at forward distance height / normalized_y.
    const double forward_distance = 1.0 / lowest_point->y;
    const double lateral_distance = -lowest_point->x * forward_distance;
    EXPECT_NEAR(cluster.camera_relative_translation->X().value(),
                forward_distance, 1e-5);
    EXPECT_NEAR(cluster.camera_relative_translation->Y().value(),
                lateral_distance, 1e-5);
    EXPECT_GE(cluster.centroid.y, 0.12f);
    EXPECT_LT(cluster.centroid.y, 0.32f);
  }
}

TEST(HSVClusterTrackerTest, NormalizedPointsReprojectWithDistortion) {
  const cv::Matx33d intrinsics{140, 0, 64, 0, 90, 48, 0, 0, 1};
  const cv::Vec<double, 5> distortion{0.1, -0.03, 0.01, -0.02, 0.005};
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", intrinsics,
                                         kExtrinsics, distortion);
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
                                         kExtrinsics, kDistortion);
  tracker.ProcessFrame(bgr);
  ASSERT_EQ(tracker.GetClusters()->size(), 1U);
  const auto& cluster = tracker.GetClusters()->front();
  ASSERT_EQ(cluster.img_points.size(), 1U);
  EXPECT_EQ(cluster.img_points.front(), cv::Point2f(0, 0.22f));
  EXPECT_EQ(cluster.centroid, cv::Point2f(0, 0.22f));
  hsv(70, 64) = cv::Vec3b{0, 0, 0};
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  tracker.ProcessFrame(bgr);
  EXPECT_TRUE(tracker.GetClusters()->empty());
}

TEST(HSVClusterTrackerTest, AppliesConfigurableGroundDistanceLimit) {
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion, 2.0f);
  tracker.ProcessFrame(MakeImage());
  EXPECT_TRUE(tracker.GetClusters()->empty());
  EXPECT_THROW(gamepiece::HSVClusterTrackerNode("decoded", "clusters",
                                               kIntrinsics, kExtrinsics,
                                               kDistortion, 0.0f),
               std::invalid_argument);
}

TEST(HSVClusterTrackerTest, ClusteringIsIndependentOfPixelOrigin) {
  const cv::Mat image = MakeImage();
  cv::Mat3b shifted_image(144, 192, cv::Vec3b{0, 0, 0});
  image.copyTo(shifted_image(cv::Rect{64, 48, image.cols, image.rows}));
  const cv::Matx33d intrinsics{128, 0, 64, 0, 128, 48, 0, 0, 1};
  const cv::Matx33d shifted_intrinsics{128, 0, 128, 0, 128, 96, 0, 0, 1};
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", intrinsics,
                                         kExtrinsics, kDistortion);
  gamepiece::HSVClusterTrackerNode shifted_tracker(
      "decoded", "clusters", shifted_intrinsics, kExtrinsics, kDistortion);
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
                                        kExtrinsics, kDistortion);
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
                                        kExtrinsics, kDistortion);
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", kIntrinsics,
                                         kExtrinsics, kDistortion);
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
      EXPECT_EQ(actual.centroid, expected.centroid);
      EXPECT_EQ(actual.img_points, expected.img_points);
      EXPECT_DOUBLE_EQ(cv::norm(actual.covar - expected.covar), 0);
      EXPECT_EQ(actual.camera_relative_translation,
                expected.camera_relative_translation);
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
