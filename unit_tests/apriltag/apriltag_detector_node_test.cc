#include <memory>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

#include "apriltag/apriltag_detector.h"
#include "apriltag/gpu_apriltag_detector_node.h"
#include "apriltag/opencv_apriltag_detector_node.h"
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "unit_tests/test_helpers.h"

namespace {

auto CudaDeviceCount() -> std::optional<int> {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    std::cerr << "WARNING: skipping GPUApriltagDetectorNode test because CUDA "
              << "device discovery failed: " << cudaGetErrorString(status)
              << '\n';
    return std::nullopt;
  }
  if (device_count <= 0) {
    std::cerr << "WARNING: skipping GPUApriltagDetectorNode test because no "
              << "CUDA-capable GPU is present\n";
    return std::nullopt;
  }
  return device_count;
}

TEST(ApriltagDetectorInterfacesTest, ConcreteDetectorsImplementInterface) {
  EXPECT_TRUE((std::is_base_of_v<apriltag::IApriltagDetectorNode,
                                 apriltag::OpenCVApriltagDetectorNode>));
  EXPECT_TRUE((std::is_base_of_v<apriltag::IApriltagDetectorNode,
                                 apriltag::GPUApriltagDetectorNode>));
}

TEST(OpenCVApriltagDetectorNodeTest, NullFramePublishesEmptyDetections) {
  apriltag::OpenCVApriltagDetectorNode detector(
      cos_test::testing::IntrinsicsJson());

  int callback_count = 0;
  control_loops::MetaDataList observed_metadata;
  detector.RegisterCallback(
      [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context> ctx) {
        callback_count++;
        observed_metadata = std::move(metadata);
        EXPECT_NE(detections, nullptr);
        EXPECT_TRUE(detections->empty());
        EXPECT_EQ(ctx, nullptr);
      });

  detector.Detect(nullptr, {{.camera_idx = 2, .timestamp = 1234}}, nullptr);

  EXPECT_EQ(callback_count, 1);
  ASSERT_EQ(observed_metadata.size(), 1);
  EXPECT_EQ(observed_metadata.front().camera_idx, 2);
  EXPECT_EQ(observed_metadata.front().timestamp, 1234UL);
}

TEST(GPUApriltagDetectorNodeTest, NullFramePublishesEmptyDetectionsWhenGpuPresent) {
  const std::optional<int> device_count = CudaDeviceCount();
  if (!device_count.has_value()) {
    GTEST_SKIP() << "WARNING: no usable CUDA GPU is present";
  }

  apriltag::GPUApriltagDetectorNode detector(
      640, 480, cos_test::testing::IntrinsicsJson());

  int callback_count = 0;
  control_loops::MetaDataList observed_metadata;
  detector.RegisterCallback(
      [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context> ctx) {
        callback_count++;
        observed_metadata = std::move(metadata);
        EXPECT_NE(detections, nullptr);
        EXPECT_TRUE(detections->empty());
        EXPECT_EQ(ctx, nullptr);
      });

  detector.Detect(nullptr, {{.camera_idx = 4, .timestamp = 5678}}, nullptr);

  EXPECT_EQ(callback_count, 1);
  ASSERT_EQ(observed_metadata.size(), 1);
  EXPECT_EQ(observed_metadata.front().camera_idx, 4);
  EXPECT_EQ(observed_metadata.front().timestamp, 5678UL);
}

TEST(TagDetectionTest, StreamsIdAndCorners) {
  apriltag::tag_detection_t detection{
      .tag_id = 7,
      .corners = {cv::Point2d{1, 2}, cv::Point2d{3, 4}, cv::Point2d{5, 6},
                  cv::Point2d{7, 8}},
      .confidence = 0.5};

  std::ostringstream out;
  out << detection;

  EXPECT_NE(out.str().find("ID: 7"), std::string::npos);
  EXPECT_NE(out.str().find("(1, 2)"), std::string::npos);
}

}  // namespace
