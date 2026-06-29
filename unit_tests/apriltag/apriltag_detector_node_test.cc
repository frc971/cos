#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#include "apriltag/apriltag_detector.h"
#include "apriltag/gpu_apriltag_detector_node.h"
#include "apriltag/opencv_apriltag_detector_node.h"
#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "unit_tests/apriltag_test_helpers.h"
#include "unit_tests/cuda_test_helpers.h"
#include "unit_tests/test_helpers.h"

namespace {

auto ApriltagArtifactRoot() -> std::filesystem::path {
  return std::filesystem::path(COS_BINARY_DIR) / "test_artifacts" /
         "apriltag_detections";
}

void WriteDetectionOverlay(
    const cv::Mat& gray,
    const std::vector<apriltag::tag_detection_t>& detections,
    const std::filesystem::path& output_path) {
  std::filesystem::create_directories(output_path.parent_path());

  cv::Mat overlay;
  cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
  for (const apriltag::tag_detection_t& detection : detections) {
    std::vector<cv::Point> corners;
    corners.reserve(detection.corners.size());
    for (const cv::Point2d& corner : detection.corners) {
      corners.emplace_back(static_cast<int>(std::lround(corner.x)),
                           static_cast<int>(std::lround(corner.y)));
    }
    cv::polylines(overlay, corners, true, cv::Scalar(0, 255, 0), 3,
                  cv::LINE_AA);
    for (size_t i = 0; i < corners.size(); ++i) {
      cv::circle(overlay, corners[i], 7, cv::Scalar(0, 0, 255), -1,
                 cv::LINE_AA);
      cv::putText(overlay, std::to_string(i), corners[i] + cv::Point{8, -8},
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2,
                  cv::LINE_AA);
    }
    if (!corners.empty()) {
      cv::putText(overlay, "id " + std::to_string(detection.tag_id),
                  corners.front() + cv::Point{12, 24},
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2,
                  cv::LINE_AA);
    }
  }

  ASSERT_TRUE(cv::imwrite(output_path.string(), overlay))
      << "Failed to write AprilTag detection proof: " << output_path;
  std::cout << "AprilTag detection proof written to " << output_path << '\n';
}

void ExpectNullFramePublishesEmptyDetections(
    std::unique_ptr<apriltag::IApriltagDetectorNode> detector,
    control_loops::MetaData expected_metadata) {
  int callback_count = 0;
  control_loops::MetaDataList observed_metadata;
  detector->RegisterCallback(
      [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context> ctx) {
        callback_count++;
        observed_metadata = std::move(metadata);
        EXPECT_NE(detections, nullptr);
        EXPECT_TRUE(detections->empty());
        EXPECT_EQ(ctx, nullptr);
      });

  detector->Detect(nullptr, {expected_metadata}, nullptr);

  EXPECT_EQ(callback_count, 1);
  ASSERT_EQ(observed_metadata.size(), 1);
  EXPECT_EQ(observed_metadata.front().camera_idx,
            expected_metadata.camera_idx);
  EXPECT_EQ(observed_metadata.front().timestamp, expected_metadata.timestamp);
}

void ExpectDetectsTwoTagsInRealLog181Frame(
    std::unique_ptr<apriltag::IApriltagDetectorNode> detector,
    const std::filesystem::path& proof_path) {
  cv::Mat gray = cos_test::testing::LoadRealApriltagFrame();
  ASSERT_FALSE(gray.empty());

  auto frame = cos_test::testing::MakeGrayNvBufferFrame(gray);
  std::shared_ptr<std::vector<apriltag::tag_detection_t>> observed_detections;
  detector->RegisterCallback(
      [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList,
          std::shared_ptr<control_loops::Context>) {
        observed_detections = std::move(detections);
      });

  detector->Detect(frame, {{.camera_idx = 0, .timestamp = 20283378}}, nullptr);

  ASSERT_NE(observed_detections, nullptr);
  WriteDetectionOverlay(gray, *observed_detections, proof_path);
  EXPECT_EQ(observed_detections->size(), 2);
}

auto MakeOpenCVDetector() -> std::unique_ptr<apriltag::IApriltagDetectorNode> {
  return std::make_unique<apriltag::OpenCVApriltagDetectorNode>(
      cos_test::testing::IntrinsicsJson());
}

auto MakeGPUDetectorForRealFrame()
    -> std::unique_ptr<apriltag::IApriltagDetectorNode> {
  cv::Mat gray = cos_test::testing::LoadRealApriltagFrame();
  EXPECT_FALSE(gray.empty());
  return std::make_unique<apriltag::GPUApriltagDetectorNode>(
      static_cast<uint>(gray.cols), static_cast<uint>(gray.rows),
      cos_test::testing::IntrinsicsJson());
}

TEST(ApriltagDetectorInterfacesTest, ConcreteDetectorsImplementInterface) {
  EXPECT_TRUE((std::is_base_of_v<apriltag::IApriltagDetectorNode,
                                 apriltag::OpenCVApriltagDetectorNode>));
  EXPECT_TRUE((std::is_base_of_v<apriltag::IApriltagDetectorNode,
                                 apriltag::GPUApriltagDetectorNode>));
}

TEST(OpenCVApriltagDetectorNodeTest, NullFramePublishesEmptyDetections) {
  ExpectNullFramePublishesEmptyDetections(
      MakeOpenCVDetector(), {.camera_idx = 2, .timestamp = 1234});
}

TEST(OpenCVApriltagDetectorNodeTest, DetectsTwoTagsInRealLog181Frame) {
  ExpectDetectsTwoTagsInRealLog181Frame(
      MakeOpenCVDetector(), ApriltagArtifactRoot() / "opencv_20.283378.png");
}

TEST(GPUApriltagDetectorNodeTest,
     DetectsTwoTagsInRealLog181FrameAndWritesVisualProof) {
  const std::optional<int> device_count = cos_test::testing::CudaDeviceCount();
  if (!device_count.has_value()) {
    GTEST_SKIP() << "WARNING: no usable CUDA GPU is present";
  }

  ExpectDetectsTwoTagsInRealLog181Frame(
      MakeGPUDetectorForRealFrame(),
      ApriltagArtifactRoot() / "gpu_20.283378.png");
}

TEST(GPUApriltagDetectorNodeTest,
     NullFramePublishesEmptyDetectionsWhenGpuPresent) {
  const std::optional<int> device_count = cos_test::testing::CudaDeviceCount();
  if (!device_count.has_value()) {
    GTEST_SKIP() << "WARNING: no usable CUDA GPU is present";
  }

  ExpectNullFramePublishesEmptyDetections(
      std::make_unique<apriltag::GPUApriltagDetectorNode>(
          640, 480, cos_test::testing::IntrinsicsJson()),
      {.camera_idx = 4, .timestamp = 5678});
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
