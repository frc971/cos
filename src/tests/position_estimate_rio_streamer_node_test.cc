#include "streamer/position_estimate_rio_streamer_node.h"

#include <chrono>
#include <memory>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>
#include <networktables/NetworkTableInstance.h>

#include "localization/position.h"
#include "localization/variance_calculator_node.h"

namespace {

void CheckSample(nt::DoubleArraySubscriber& subscriber) {
  const auto samples = subscriber.ReadQueue();
  ASSERT_EQ(samples.size(), 1);
  const std::vector<double> expected{1, 2, 0.5, 1.525, 0};
  ASSERT_EQ(samples.front().value.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(samples.front().value[i], expected[i]);
  }
}

TEST(PositionEstimateRioStreamerNodeTest, PublishesMainPoseAndTimestampFormat) {
  auto instance = nt::NetworkTableInstance::GetDefault();
  auto subscriber = instance.GetDoubleArrayTopic("/COS/PositionEstimate")
                        .Subscribe({}, {.pollStorage = 10,
                                        .sendAll = true,
                                        .keepDuplicates = true});
  auto pose2d = instance.GetStructTopic<frc::Pose2d>("/COS/Pose2d").Subscribe({});
  auto pose3d = instance.GetStructTopic<frc::Pose3d>("/COS/Pose3d").Subscribe({});
  streamer::PositionEstimateRioStreamerNode sender("pose_with_variance", "/COS");
  localization::VarianceCalculatorNode variance("pose", "pose_with_variance");
  variance.RegisterCallback(sender.CreateCallback());
  int callback_count = 0;
  sender.RegisterCallback(
      [&callback_count](const control_loop::Context&) -> void {
        ++callback_count;
      });
  subscriber.ReadQueue();

  const frc::Pose3d pose{
      units::meter_t{1}, units::meter_t{2}, units::meter_t{3},
      frc::Rotation3d{units::radian_t{0}, units::radian_t{0},
                      units::radian_t{0.5}}};
  const auto publish = variance.CreateCallback();
  for (int i = 0; i < 2; ++i) {
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, i);
    auto estimate = std::make_unique<localization::PositionEstimateMessage>();
    estimate->pose = pose;
    estimate->tag_ids = {1, 2};
    estimate->distances = {2, 4};
    // Main derives the timestamp from registered cameras, ignoring this field.
    // With no cameras registered, it publishes zero.
    estimate->timestamp = 12.5;
    context->SetMessage("pose", std::move(estimate));
    publish(context);
    CheckSample(subscriber);
    EXPECT_EQ(pose2d.Get(), pose.ToPose2d());
    EXPECT_EQ(pose3d.Get(), pose);
  }
  EXPECT_EQ(instance.GetTopic("/COS/PositionEstimate").GetTypeString(),
            "double[]");

  auto context = std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 2);
  context->SetMessage("pose", nullptr);
  publish(context);
  EXPECT_TRUE(subscriber.ReadQueue().empty());
  sender.CreateCallback()(std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 3));
  EXPECT_TRUE(subscriber.ReadQueue().empty());
  EXPECT_EQ(callback_count, 4);
}

}  // namespace
