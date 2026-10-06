#include "localization/square_solver_node.h"
#include "localization/unambiguous_solver_node.h"
#include "utils/cv_geometry.h"
#include "camera/jpeg_buffer.h"
#include "control_loop/control_loop.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <gtest/gtest.h>

namespace {

auto FrontConfig() -> std::filesystem::path {
  const char* constants_dir = std::getenv("COS_TEST_CONSTANTS_DIR");
  return std::filesystem::path(constants_dir ? constants_dir
                                             : COS_TEST_CONSTANTS_DIR) /
         "vision-bot/front.json";
}

class SingleTagRotationTest : public ::testing::Test {
 protected:
  camera::Intrinsics intrinsics_{FrontConfig()};
  camera::Extrinsics extrinsics_{FrontConfig()};
  frc::AprilTagFieldLayout layout_{
      {{10, frc::Pose3d{0_m, 0_m, 0_m,
                        frc::Rotation3d{0_rad, 0_rad, -90_deg}}}}, 20_m, 10_m};
  localization::SquareSolverNode square_{"detections", "candidates",
                                          intrinsics_, extrinsics_, layout_};
  localization::UnambiguousSolverNode solver_{"pose", layout_};

  auto Detection(double yaw_degrees) -> localization::tag_detection_t {
    // With this tag's field heading, a CV Y rotation of yaw - 90 produces
    // the requested robot yaw. Use real calibrated projection/distortion.
    const cv::Mat yaw_rvec = (cv::Mat_<double>(3, 1) <<
        0, (yaw_degrees - 90) * std::numbers::pi / 180, 0);
    // A slight fixed pitch avoids IPPE's exactly frontoparallel singularity.
    const cv::Mat pitch_rvec = (cv::Mat_<double>(3, 1) << 0.05, 0, 0);
    cv::Mat yaw_rotation, pitch_rotation, rvec;
    cv::Rodrigues(yaw_rvec, yaw_rotation);
    cv::Rodrigues(pitch_rvec, pitch_rotation);
    cv::Rodrigues(pitch_rotation * yaw_rotation, rvec);
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.03, 0, 0.6);
    std::vector<cv::Point2d> corners;
    cv::projectPoints(localization::kApriltagCorners, rvec, tvec,
                      intrinsics_.ToMatrix(),
                      intrinsics_.ToDistortionCoefficients(), corners);
    localization::tag_detection_t detection;
    detection.tag_id = 10;
    std::copy(corners.begin(), corners.end(), detection.corners.begin());
    return detection;
  }
};

TEST_F(SingleTagRotationTest, ClearImageEvidenceOverridesMirroredPreviousPose) {
  const auto detection = Detection(135);
  std::vector<cv::Mat> rvecs, tvecs;
  cv::solvePnPGeneric(localization::kApriltagCorners, detection.corners,
                      intrinsics_.ToMatrix(),
                      intrinsics_.ToDistortionCoefficients(), rvecs, tvecs,
                      false, cv::SOLVEPNP_IPPE_SQUARE);
  ASSERT_EQ(rvecs.size(), 2u);
  localization::ambiguous_estimate_t previous;
  previous.pos1.pose = utils::ComputeRobotPose(
      tvecs[1], rvecs[1], 10, layout_, extrinsics_.ToCameraToRobot<cv::Mat>());
  previous.pos1.variance = 1;
  ASSERT_TRUE(solver_.Solve({&previous}, false).has_value());

  auto estimate = square_.AmbiguousSolve(detection, false);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_FALSE(estimate->pos2.has_value());
  const auto pose = solver_.Solve({&*estimate}, false);
  ASSERT_TRUE(pose.has_value());
  EXPECT_NEAR(units::degree_t{pose->pose.Rotation().Z()}.value(), 135, 0.01);
}

TEST_F(SingleTagRotationTest, YawContinuesThroughNinetyInBothDirections) {
  for (const int direction : {1, -1}) {
    localization::UnambiguousSolverNode solver{"pose", layout_};
    for (int step = 0; step <= 160; ++step) {
      const double yaw = direction == 1 ? 10 + step : 170 - step;
      SCOPED_TRACE(::testing::Message() << "direction=" << direction
                                      << " yaw=" << yaw);
      auto estimate = square_.AmbiguousSolve(Detection(yaw), false);
      ASSERT_TRUE(estimate.has_value());
      const auto pose = solver.Solve({&*estimate}, false);
      ASSERT_TRUE(pose.has_value());
      EXPECT_NEAR(units::degree_t{pose->pose.Rotation().Z()}.value(), yaw, 0.1)
          << "direction=" << direction << " requested yaw=" << yaw;
    }
  }
}

TEST_F(SingleTagRotationTest, RetainsBothCandidatesWhenImageFitIsAmbiguous) {
  auto detection = Detection(90);
  for (int i = 0; i < 4; ++i) {
    detection.corners[i].x += i % 2 == 0 ? 1.0 : -1.0;
  }
  auto estimate = square_.AmbiguousSolve(detection, false);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_TRUE(estimate->pos2.has_value());
}

TEST_F(SingleTagRotationTest, PublishesMeanCaptureTimeWithThePose) {
  control_loop::ControlLoop loop(std::chrono::milliseconds(1));
  solver_.SetRejectFarTags(false);
  solver_.AddCamera("detections", intrinsics_, extrinsics_, loop);
  for (const auto* channel : {"jpeg/first", "jpeg/second", "jpeg/invalid",
                             "jpeg/missing"}) {
    solver_.AddCameraTimestamp(channel);
  }
  const auto callback = solver_.CreateCallback();
  for (bool have_frames : {true, false}) {
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
    localization::AmbiguousEstimate candidates;
    candidates.pos1.pose = frc::Pose3d{1_m, 2_m, 0_m, frc::Rotation3d{}};
    candidates.pos1.variance = 1;
    context->SetMessage("detections:multitag_solver",
        std::make_unique<localization::AmbiguousEstimateMessage>(candidates));
    if (have_frames) {
      context->SetMessage("jpeg/first",
                         std::make_unique<camera::JpegBuffer>(0, 12));
      context->SetMessage("jpeg/second",
                         std::make_unique<camera::JpegBuffer>(0, 16));
      context->SetMessage("jpeg/invalid",
          std::make_unique<camera::JpegBuffer>(
              0, std::numeric_limits<double>::quiet_NaN()));
    }
    context->SetMessage("jpeg/missing", nullptr);
    callback(context);
    const auto* output =
        context->GetMessage<localization::PositionEstimateMessage>("pose");
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->pose, candidates.pos1.pose);
    EXPECT_DOUBLE_EQ(output->timestamp, have_frames ? 14 : 0);
  }
}

}  // namespace
