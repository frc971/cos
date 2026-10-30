#include <chrono>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "camera/jpeg_buffer.h"
#include "streamer/ball_pathing_streamer_node.h"

namespace {

class BallPathingOverlayTest : public testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
        ("ball-overlay-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory_);
  }
  void TearDown() override { std::filesystem::remove_all(directory_); }

  auto Config(double yaw = 0, bool distorted = false) -> std::string {
    const auto path = directory_ / "camera.json";
    std::ofstream output(path);
    output << R"({"intrinsics":{"fx":100,"fy":100,"cx":200,"cy":150,)"
           << "\"k1\":" << (distorted ? 0.1 : 0) << ','
           << "\"k2\":" << (distorted ? 0.01 : 0) << ','
           << "\"k3\":" << (distorted ? 0.001 : 0) << ','
           << "\"p1\":" << (distorted ? 0.002 : 0) << ','
           << "\"p2\":" << (distorted ? -0.003 : 0) << "},"
           << R"("extrinsics":{"translation_x":0,"translation_y":0,)"
           << R"("translation_z":1,"rotation_x":0,"rotation_y":0,"rotation_z":)"
           << yaw << "}}";
    return path.string();
  }

  auto Goal() -> gamepiece::BallVelocityGoalMessage {
    gamepiece::BallVelocityGoalMessage goal;
    goal.valid = true;
    goal.target_x_m = 2;
    goal.vx_mps = 1;
    return goal;
  }

  std::filesystem::path directory_;
};

TEST_F(BallPathingOverlayTest, ProjectsFloorUsingCameraHeightAndRobotAxes) {
  streamer::BallPathingOverlay overlay(Config(), 0.4);
  const auto center = overlay.ProjectFloorPoint({2, 0});
  ASSERT_TRUE(center);
  EXPECT_NEAR(center->x, 200, 1e-9);
  EXPECT_NEAR(center->y, 200, 1e-9);
  const auto left = overlay.ProjectFloorPoint({2, 0.5});
  ASSERT_TRUE(left);
  EXPECT_NEAR(left->x, 175, 1e-9);
  EXPECT_NEAR(left->y, 200, 1e-9);
  EXPECT_FALSE(overlay.ProjectFloorPoint({-2, 0}));
  EXPECT_FALSE(overlay.ProjectFloorPoint({0, 0}));
}

TEST_F(BallPathingOverlayTest, HonorsMountRotationInDegrees) {
  streamer::BallPathingOverlay overlay(Config(90), 0.4);
  const auto point = overlay.ProjectFloorPoint({0, 2});
  ASSERT_TRUE(point);
  EXPECT_NEAR(point->x, 200, 1e-9);
  EXPECT_NEAR(point->y, 200, 1e-9);
  EXPECT_FALSE(overlay.ProjectFloorPoint({0, -2}));
}

TEST_F(BallPathingOverlayTest, DistortionMatchesOpenCvProjection) {
  streamer::BallPathingOverlay overlay(Config(0, true), 0.4);
  const auto point = overlay.ProjectFloorPoint({2, 0.5});
  ASSERT_TRUE(point);
  std::vector<cv::Point2d> expected;
  cv::projectPoints(std::vector<cv::Point3d>{{-0.5, 1, 2}}, cv::Vec3d{},
                    cv::Vec3d{}, cv::Matx33d(100, 0, 200, 0, 100, 150, 0, 0, 1),
                    cv::Vec<double, 5>(0.1, 0.01, 0.002, -0.003, 0.001), expected);
  EXPECT_NEAR(point->x, expected[0].x, 1e-9);
  EXPECT_NEAR(point->y, expected[0].y, 1e-9);
}

TEST_F(BallPathingOverlayTest, DrawsTargetWithoutModifyingInput) {
  streamer::BallPathingOverlay overlay(Config(), 0.4);
  const cv::Mat image = cv::Mat::zeros(300, 400, CV_8UC3);
  const auto original = image.clone();
  auto goal = Goal();
  const auto result = overlay.Render(image, &goal);
  EXPECT_EQ(result.at<cv::Vec3b>(200, 200), cv::Vec3b(210, 40, 255));
  EXPECT_EQ(cv::norm(image, original), 0);
  EXPECT_GT(cv::norm(result, image), 0);
}

TEST_F(BallPathingOverlayTest, RotatesTargetAndImageTogether) {
  streamer::BallPathingOverlay overlay(Config(), 0.4, true);
  cv::Mat image = cv::Mat::zeros(300, 400, CV_8UC3);
  image.at<cv::Vec3b>(20, 300) = {10, 20, 30};
  auto goal = Goal();
  const auto result = overlay.Render(image, &goal);
  EXPECT_EQ(result.at<cv::Vec3b>(99, 199), cv::Vec3b(210, 40, 255));
  // This image pixel rotates to (99,279), below the annotations.
  EXPECT_EQ(result.at<cv::Vec3b>(279, 99), cv::Vec3b(10, 20, 30));
}

TEST_F(BallPathingOverlayTest, DistinguishesHeldAndInvalidGoals) {
  streamer::BallPathingOverlay overlay(Config(), 0.4);
  const cv::Mat image = cv::Mat::zeros(300, 400, CV_8UC3);
  auto goal = Goal();
  const auto tracking = overlay.Render(image, &goal);
  goal.held = true;
  const auto held = overlay.Render(image, &goal);
  EXPECT_GT(cv::norm(held, tracking), 0);
  EXPECT_EQ(held.at<cv::Vec3b>(200, 200), cv::Vec3b(210, 40, 255));
  goal.valid = false;
  const auto invalid = overlay.Render(image, &goal);
  EXPECT_EQ(invalid.at<cv::Vec3b>(200, 200), cv::Vec3b(0, 0, 0));
  EXPECT_EQ(cv::norm(invalid, overlay.Render(image, nullptr)), 0);
}

TEST_F(BallPathingOverlayTest, ClipsOffScreenAndBehindCameraCircleSegments) {
  streamer::BallPathingOverlay overlay(Config(), 0.4);
  const cv::Mat image = cv::Mat::zeros(300, 400, CV_8UC3);
  auto goal = Goal();
  for (const cv::Point2d target : {cv::Point2d(0.1, 5), cv::Point2d(-2, 0),
                                  cv::Point2d(2, 1e9)}) {
    goal.target_x_m = target.x;
    goal.target_y_m = target.y;
    const auto result = overlay.Render(image, &goal);
    EXPECT_EQ(result.size(), image.size());
    EXPECT_GT(cv::norm(result, image), 0);
  }
}

TEST_F(BallPathingOverlayTest, CompletesContextCallbacksBeforeRendering) {
  const auto caller = std::this_thread::get_id();
  std::atomic<int> callbacks{0};
  std::atomic<bool> wrong_thread{false};
  {
    // Port 0 requests an ephemeral listening port for this test.
    streamer::BallPathingStreamerNode node("jpeg", "goal", Config(), 0.4,
                                           "/stream", 0);
    EXPECT_EQ(node.GetDependencies().size(), 2U);
    EXPECT_TRUE(node.GetPublications().empty());
    node.RegisterCallback([&](const control_loop::Context&) {
      if (std::this_thread::get_id() != caller) wrong_thread.store(true);
      ++callbacks;
    });
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
    std::vector<unsigned char> encoded;
    ASSERT_TRUE(cv::imencode(".jpg", cv::Mat::zeros(800, 1280, CV_8UC3), encoded));
    auto jpeg = std::make_shared<camera::JpegBuffer>(encoded.size(), 1.0);
    std::memcpy(jpeg->ptr, encoded.data(), encoded.size());
    context->SetMessage("jpeg", jpeg);
    context->SetMessage("goal",
                        std::make_shared<gamepiece::BallVelocityGoalMessage>(Goal()));
    const auto callback = node.CreateCallback();
    callback(context);
    EXPECT_EQ(callbacks.load(), 1);
    auto missing = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 2);
    callback(missing);
    EXPECT_EQ(callbacks.load(), 2);
  }
  EXPECT_FALSE(wrong_thread.load());
  EXPECT_EQ(callbacks.load(), 2);
}

}  // namespace
