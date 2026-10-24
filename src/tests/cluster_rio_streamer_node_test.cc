#include "streamer/cluster_rio_streamer_node.h"

#include <chrono>
#include <limits>
#include <memory>
#include <numbers>
#include <stop_token>
#include <thread>

#include <gtest/gtest.h>
#include <networktables/RawTopic.h>

#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/position.h"

namespace {

class ClusterSenderTest : public ::testing::Test {
 protected:
  nt::NetworkTableInstance instance_ = nt::NetworkTableInstance::Create();
  void TearDown() override { nt::NetworkTableInstance::Destroy(instance_); }
  static auto Context() -> control_loop::Context {
    return std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
  }
  static auto Cluster() -> gamepiece::kmeans_cluster_t {
    gamepiece::kmeans_cluster_t cluster;
    cluster.robot_relative_translation =
        frc::Translation2d{units::meter_t{2}, units::meter_t{-1}};
    cluster.ground_covar = {9, 0, 0, 4};
    cluster.density = 7.25f;
    return cluster;
  }
  static void SetPose(const control_loop::Context& context) {
    auto estimate = std::make_shared<localization::PositionEstimateMessage>();
    estimate->pose = frc::Pose3d{
        units::meter_t{10}, units::meter_t{-3}, units::meter_t{0},
        frc::Rotation3d{units::radian_t{0}, units::radian_t{0},
                        units::radian_t{std::numbers::pi / 2}}};
    context->SetMessage("pose", estimate);
  }
  static void SetClusters(const control_loop::Context& context,
                          gamepiece::hsv_clusters_t clusters) {
    context->SetMessage(
        "clusters", std::make_shared<
                        control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(
                        std::move(clusters)));
  }
};

TEST_F(ClusterSenderTest, JoinsEitherArrivalOrderAndSendsOnceConcurrently) {
  streamer::ClusterRioStreamerNode sender("clusters", "pose", "/COS/Clusters",
                                          instance_);
  auto subscriber = instance_.GetStructArrayTopic<gamepiece::cluster_nt_t>(
                                 "/COS/Clusters")
                        .Subscribe({}, {.pollStorage = 20,
                                        .sendAll = true,
                                        .keepDuplicates = true});
  int callbacks = 0;
  sender.RegisterCallback([&](const auto&) -> void { ++callbacks; });
  auto callback = sender.CreateCallback();
  for (bool pose_first : {false, true}) {
    auto context = Context();
    if (pose_first) SetPose(context);
    else SetClusters(context, {Cluster()});
    callback(context);
    EXPECT_TRUE(subscriber.ReadQueue().empty());
    if (pose_first) SetClusters(context, {Cluster()});
    else SetPose(context);
    std::thread first([&] -> void { callback(context); });
    std::thread second([&] -> void { callback(context); });
    first.join();
    second.join();
    const auto samples = subscriber.ReadQueue();
    ASSERT_EQ(samples.size(), 1);
    ASSERT_EQ(samples[0].value.size(), 1);
    const auto& value = samples[0].value[0];
    EXPECT_NEAR(value.center_x, 11, 1e-9);
    EXPECT_NEAR(value.center_y, -1, 1e-9);
    EXPECT_NEAR(value.field_relative_semimajor_axis_yaw, std::numbers::pi / 2,
                1e-9);
    EXPECT_DOUBLE_EQ(value.semimajor_length, 6);
    EXPECT_DOUBLE_EQ(value.semiminor_length, 4);
    EXPECT_DOUBLE_EQ(value.predicted_density, 7.25);
  }
  EXPECT_EQ(callbacks, 2);
  EXPECT_EQ(subscriber.GetTopic().GetTypeString(), "struct:ClusterNT[]");
}

TEST_F(ClusterSenderTest, HandlesRotatedDegenerateAndInvalidCovariance) {
  streamer::ClusterRioStreamerNode sender("clusters", "pose", "/COS/Clusters",
                                          instance_);
  auto context = Context();
  SetPose(context);
  auto rotated = Cluster();
  rotated.ground_covar = {5, 4, 4, 5};
  auto degenerate = Cluster();
  degenerate.ground_covar = cv::Matx22f::zeros();
  auto invalid = Cluster();
  invalid.ground_covar = {1, 2, 2, 1};
  auto nan = Cluster();
  nan.density = std::numeric_limits<float>::quiet_NaN();
  auto missing = Cluster();
  missing.robot_relative_translation.reset();
  SetClusters(context, {rotated, degenerate, invalid, nan, missing});
  sender.CreateCallback()(context);
  const auto* output = context->GetMessage<
      control_loop::ValueMessage<std::vector<gamepiece::cluster_nt_t>>>(
      "/COS/Clusters");
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->value.size(), 2);
  EXPECT_NEAR(output->value[0].field_relative_semimajor_axis_yaw,
              3 * std::numbers::pi / 4, 1e-9);
  EXPECT_DOUBLE_EQ(output->value[0].semimajor_length, 6);
  EXPECT_DOUBLE_EQ(output->value[0].semiminor_length, 2);
  EXPECT_DOUBLE_EQ(output->value[1].semimajor_length, 0);
  EXPECT_DOUBLE_EQ(output->value[1].semiminor_length, 0);
}

TEST_F(ClusterSenderTest, EmptyAndMissingPoseClearTargetsButNullFrameDoesNotSend) {
  streamer::ClusterRioStreamerNode sender("clusters", "pose", "/COS/Clusters",
                                          instance_);
  // WPILib 2026's typed C++ queue reader discards zero-length struct arrays.
  // Verify the wire values directly so empty observations remain observable.
  auto subscriber = instance_.GetRawTopic("/COS/Clusters")
                        .Subscribe("struct:ClusterNT[]", {},
                                   {.sendAll = true, .keepDuplicates = true});
  auto callback = sender.CreateCallback();
  for (bool missing_pose : {false, true}) {
    auto context = Context();
    if (missing_pose) context->SetMessage("pose", nullptr);
    else SetPose(context);
    SetClusters(context, missing_pose ? gamepiece::hsv_clusters_t{Cluster()}
                                     : gamepiece::hsv_clusters_t{});
    callback(context);
    auto queue = subscriber.ReadQueue();
    ASSERT_EQ(queue.size(), 1);
    EXPECT_TRUE(queue[0].value.empty());
  }
  auto context = Context();
  SetPose(context);
  context->SetMessage("clusters", nullptr);
  callback(context);
  EXPECT_TRUE(subscriber.ReadQueue().empty());
  EXPECT_TRUE(context->Exists("/COS/Clusters"));
}

}  // namespace
