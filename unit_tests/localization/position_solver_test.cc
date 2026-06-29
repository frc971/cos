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
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position_solver.h"
#include "localization/square_solver_node.h"
#include "localization/unambiguous_solver_node.h"
#include "opencv2/imgcodecs.hpp"
#include "unit_tests/test_helpers.h"

namespace {

auto SolverFixtureRoot() -> std::filesystem::path {
  return std::filesystem::path(COS_BINARY_DIR) / "test_artifacts" /
         "position_solver";
}

auto RealApriltagFramePath() -> std::filesystem::path {
  return std::filesystem::path(COS_SOURCE_DIR) / "unit_tests" / "testdata" /
         "jpeg_frames" / "decoded" / "20.283378.png";
}

auto CudaDeviceCount() -> std::optional<int> {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    return std::nullopt;
  }
  return device_count;
}

auto LoadRealApriltagFrame() -> cv::Mat {
  const std::filesystem::path frame_path = RealApriltagFramePath();
  cv::Mat gray = cv::imread(frame_path.string(), cv::IMREAD_GRAYSCALE);
  EXPECT_FALSE(gray.empty()) << "Missing decoded AprilTag fixture: "
                             << frame_path;
  EXPECT_TRUE(gray.empty() || gray.isContinuous());
  return gray;
}

auto MakeGrayNvBufferFrame(const cv::Mat& gray)
    -> std::shared_ptr<camera::DecodedJpegNvBuffer> {
  auto* nv_buffer =
      new NvBuffer(V4L2_PIX_FMT_GREY, static_cast<uint32_t>(gray.cols),
                   static_cast<uint32_t>(gray.rows), 0);
  nv_buffer->planes[0].data = gray.data;
  nv_buffer->planes[0].bytesused =
      static_cast<uint32_t>(gray.total() * gray.elemSize());
  nv_buffer->planes[0].length = nv_buffer->planes[0].bytesused;
  nv_buffer->planes[0].fmt.width = static_cast<uint32_t>(gray.cols);
  nv_buffer->planes[0].fmt.height = static_cast<uint32_t>(gray.rows);
  nv_buffer->planes[0].fmt.bytesperpixel = 1;
  nv_buffer->planes[0].fmt.stride = static_cast<uint32_t>(gray.step);
  nv_buffer->planes[0].fmt.sizeimage = nv_buffer->planes[0].bytesused;
  return std::make_shared<camera::DecodedJpegNvBuffer>(nv_buffer);
}

auto DetectWith(apriltag::IApriltagDetectorNode& detector, const cv::Mat& gray)
    -> std::vector<apriltag::tag_detection_t> {
  std::shared_ptr<std::vector<apriltag::tag_detection_t>> observed_detections;
  detector.RegisterCallback(
      [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList,
          std::shared_ptr<control_loops::Context>) {
        observed_detections = std::move(detections);
      });

  detector.Detect(MakeGrayNvBufferFrame(gray),
                  {{.camera_idx = 0, .timestamp = 20283378}}, nullptr);

  EXPECT_NE(observed_detections, nullptr);
  if (observed_detections == nullptr) {
    return {};
  }
  return *observed_detections;
}

auto DetectWithBestAvailableDetector()
    -> std::vector<apriltag::tag_detection_t> {
  cv::Mat gray = LoadRealApriltagFrame();
  if (gray.empty()) {
    return {};
  }

  if (CudaDeviceCount().has_value()) {
    apriltag::GPUApriltagDetectorNode gpu_detector(
        static_cast<uint>(gray.cols), static_cast<uint>(gray.rows),
        cos_test::testing::IntrinsicsJson());
    std::vector<apriltag::tag_detection_t> detections =
        DetectWith(gpu_detector, gray);
    if (!detections.empty()) {
      std::cout << "Position solver test using GPU AprilTag detections\n";
      return detections;
    }
  }

  apriltag::OpenCVApriltagDetectorNode opencv_detector(
      cos_test::testing::IntrinsicsJson());
  std::cout << "Position solver test using OpenCV AprilTag detections\n";
  return DetectWith(opencv_detector, gray);
}

auto TranslationDistance(const wpi::math::Pose3d& a,
                         const wpi::math::Pose3d& b) -> double {
  return a.Translation().Distance(b.Translation()).value();
}

auto RotationDistance(const wpi::math::Pose3d& a,
                      const wpi::math::Pose3d& b) -> double {
  return a.Rotation().RelativeTo(b.Rotation()).Angle().value();
}

void ExpectPoseNear(const localization::position_estimate_t& actual,
                    const localization::position_estimate_t& expected,
                    double translation_margin, double rotation_margin) {
  EXPECT_NEAR(actual.pose.X().value(), expected.pose.X().value(),
              translation_margin);
  EXPECT_NEAR(actual.pose.Y().value(), expected.pose.Y().value(),
              translation_margin);
  EXPECT_NEAR(actual.pose.Z().value(), expected.pose.Z().value(),
              translation_margin);
  EXPECT_LE(TranslationDistance(actual.pose, expected.pose),
            translation_margin);
  EXPECT_LE(RotationDistance(actual.pose, expected.pose), rotation_margin);
}

auto MatchesEitherSquareEstimate(
    const localization::ambiguous_estimate_t& square_estimate,
    const localization::position_estimate_t& expected,
    double translation_margin, double rotation_margin) -> bool {
  auto matches = [&](const localization::position_estimate_t& estimate) {
    return TranslationDistance(estimate.pose, expected.pose) <=
               translation_margin &&
           RotationDistance(estimate.pose, expected.pose) <= rotation_margin;
  };
  return matches(square_estimate.pos1) ||
         (square_estimate.pos2.has_value() && matches(*square_estimate.pos2));
}

class PositionSolverFixture : public testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::create_directories(SolverFixtureRoot());
    intrinsics_path_ = cos_test::testing::WriteJsonFile(
        SolverFixtureRoot() / "solver_intrinsics_test.json",
        cos_test::testing::IntrinsicsJson());
    extrinsics_path_ = cos_test::testing::WriteJsonFile(
        SolverFixtureRoot() / "solver_extrinsics_test.json",
        cos_test::testing::ExtrinsicsJson());
  }

  std::string intrinsics_path_;
  std::string extrinsics_path_;
};

TEST(PositionEstimateTest, StreamsPoseAndSummaryFields) {
  localization::position_estimate_t estimate{
      .pose = wpi::math::Pose3d{wpi::units::meter_t{1.0},
                                wpi::units::meter_t{2.0},
                                wpi::units::meter_t{3.0},
                                wpi::math::Rotation3d{}},
      .variance = 0.25,
      .num_tags = 2};

  std::ostringstream out;
  out << estimate;

  EXPECT_NE(out.str().find("pose(x=1"), std::string::npos);
  EXPECT_NE(out.str().find("variance=0.25"), std::string::npos);
  EXPECT_NE(out.str().find("num_tags=2"), std::string::npos);
}

TEST(PositionSolverTest, VarianceScalesByDistanceAndTagCount) {
  EXPECT_DOUBLE_EQ(localization::Variance(2, 8.0, 1.0, 0.5), 2.0);
}

TEST_F(PositionSolverFixture, SolversAgreeUsingRealDetectorOutput) {
  const std::vector<apriltag::tag_detection_t> detections =
      DetectWithBestAvailableDetector();
  ASSERT_GE(detections.size(), 2);

  localization::SquareSolverNode square_solver(intrinsics_path_,
                                               extrinsics_path_);
  localization::MultiTagSolverNode multi_tag_solver(intrinsics_path_,
                                                    extrinsics_path_);
  localization::UnambiguousSolverNode unambiguous_solver(
      {cos_test::testing::MakeCameraConstant("front", intrinsics_path_,
                                             extrinsics_path_)});

  const std::vector<localization::ambiguous_estimate_t> square_estimates =
      square_solver.AmbiguousSolveWithoutNotify(detections);
  const std::optional<localization::ambiguous_estimate_t> multi_tag_estimate =
      multi_tag_solver.AmbiguousSolveWithoutNotify(detections);
  const std::optional<localization::position_estimate_t>
      unambiguous_estimate = unambiguous_solver.SolveWithoutNotify(
          {detections}, {{{.camera_idx = 0, .timestamp = 20283378}}});

  ASSERT_EQ(square_estimates.size(), detections.size());
  for (const localization::ambiguous_estimate_t& estimate : square_estimates) {
    EXPECT_TRUE(estimate.pos2.has_value());
  }
  ASSERT_TRUE(multi_tag_estimate.has_value());
  ASSERT_TRUE(unambiguous_estimate.has_value());
  ASSERT_FALSE(multi_tag_estimate->pos2.has_value());

  constexpr double kMultitagToUnambiguousTranslationMargin = 1e-9;
  constexpr double kMultitagToUnambiguousRotationMargin = 1e-9;
  ExpectPoseNear(*unambiguous_estimate, multi_tag_estimate->pos1,
                 kMultitagToUnambiguousTranslationMargin,
                 kMultitagToUnambiguousRotationMargin);

  constexpr double kSquareToMultitagTranslationMargin = 0.75;
  constexpr double kSquareToMultitagRotationMargin = 0.5;
  for (const localization::ambiguous_estimate_t& square_estimate :
       square_estimates) {
    EXPECT_TRUE(MatchesEitherSquareEstimate(
        square_estimate, multi_tag_estimate->pos1,
        kSquareToMultitagTranslationMargin, kSquareToMultitagRotationMargin))
        << "Neither square-solver branch matched the multi-tag solve";
  }
}

TEST(PositionSolverInterfacesTest, ConcreteSolversImplementInterfaces) {
  EXPECT_TRUE((std::is_base_of_v<localization::IPositionSolverNode,
                                 localization::SquareSolverNode>));
  EXPECT_TRUE((std::is_base_of_v<localization::IPositionSolverNode,
                                 localization::MultiTagSolverNode>));
  EXPECT_TRUE((std::is_base_of_v<localization::IAccumulatingSolverNode,
                                 localization::UnambiguousSolverNode>));
}

}  // namespace
