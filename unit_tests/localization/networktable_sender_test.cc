#include "localization/networktable_sender.h"
#include "gtest/gtest.h"

namespace {

constexpr int kMaxTags = 50;

TEST(NetworkTableSenderTest, SendPublishesEveryFieldToNetworkTables) {
  auto instance = wpi::nt::NetworkTableInstance::Create();
  instance.StartLocal();

  {
    const std::string camera_name = "unit-test-camera";
    auto table = instance.GetTable("Orin/PoseEstimate/" + camera_name);

    auto pose_subscriber =
        table->GetStructTopic<wpi::math::Pose2d>("Pose").Subscribe({});
    auto pose3d_subscriber =
        table->GetStructTopic<wpi::math::Pose3d>("Pose3d").Subscribe({});
    auto tag_estimation_subscriber =
        table->GetDoubleArrayTopic("TagEstimation").Subscribe({});
    auto tag_ids_subscriber =
        table->GetBooleanArrayTopic("TagId").Subscribe({});
    auto rejected_tag_ids_subscriber =
        table->GetBooleanArrayTopic("RejectedTagId").Subscribe({});
    auto num_tags_subscriber = table->GetIntegerTopic("NumTags").Subscribe(-1);
    auto variance_subscriber =
        table->GetDoubleTopic("Variance").Subscribe(-1.0);
    auto avg_tag_dist_subscriber =
        table->GetDoubleTopic("AvgTagDist").Subscribe(-1.0);
    auto loss_subscriber = table->GetDoubleTopic("Loss").Subscribe(-1.0);

    localization::NetworkTableSender sender(camera_name, instance);
    localization::position_estimate_t estimate{
        .tag_ids = {1, 49, 50},
        .rejected_tag_ids = {2, 99},
        .pose =
            wpi::math::Pose3d{wpi::units::meter_t{1.0},
                              wpi::units::meter_t{2.0},
                              wpi::units::meter_t{3.0},
                              wpi::math::Rotation3d{wpi::units::radian_t{0.1},
                                                    wpi::units::radian_t{0.2},
                                                    wpi::units::radian_t{0.3}}},
        .variance = 0.75,
        .num_tags = 3,
        .avg_tag_dist = 2.5,
        .loss = 0.1};

    sender.Send(estimate,
                {{.camera_idx = 0, .timestamp = 100},
                 {.camera_idx = 1, .timestamp = 120}},
                nullptr);
    instance.FlushLocal();

    const auto pose = pose_subscriber.Get();
    EXPECT_DOUBLE_EQ(pose.X().value(), estimate.pose.X().value());
    EXPECT_DOUBLE_EQ(pose.Y().value(), estimate.pose.Y().value());
    EXPECT_DOUBLE_EQ(pose.Rotation().Radians().value(),
                     estimate.pose.ToPose2d().Rotation().Radians().value());

    const auto pose3d = pose3d_subscriber.Get();
    EXPECT_DOUBLE_EQ(pose3d.X().value(), estimate.pose.X().value());
    EXPECT_DOUBLE_EQ(pose3d.Y().value(), estimate.pose.Y().value());
    EXPECT_DOUBLE_EQ(pose3d.Z().value(), estimate.pose.Z().value());
    EXPECT_DOUBLE_EQ(pose3d.Rotation().X().value(),
                     estimate.pose.Rotation().X().value());
    EXPECT_DOUBLE_EQ(pose3d.Rotation().Y().value(),
                     estimate.pose.Rotation().Y().value());
    EXPECT_DOUBLE_EQ(pose3d.Rotation().Z().value(),
                     estimate.pose.Rotation().Z().value());

    EXPECT_EQ(tag_estimation_subscriber.Get(),
              (std::vector<double>{
                  estimate.pose.X().value(), estimate.pose.Y().value(),
                  estimate.pose.Rotation().Z().value(), estimate.variance,
                  static_cast<double>(estimate.num_tags), estimate.avg_tag_dist,
                  estimate.loss}));

    std::vector<int> expected_tag_ids(kMaxTags);
    expected_tag_ids[1] = true;
    expected_tag_ids[49] = true;
    EXPECT_EQ(tag_ids_subscriber.Get(), expected_tag_ids);

    std::vector<int> expected_rejected_tag_ids(kMaxTags);
    expected_rejected_tag_ids[2] = true;
    EXPECT_EQ(rejected_tag_ids_subscriber.Get(), expected_rejected_tag_ids);

    EXPECT_EQ(num_tags_subscriber.Get(), estimate.num_tags);
    EXPECT_DOUBLE_EQ(variance_subscriber.Get(), estimate.variance);
    EXPECT_DOUBLE_EQ(avg_tag_dist_subscriber.Get(), estimate.avg_tag_dist);
    EXPECT_DOUBLE_EQ(loss_subscriber.Get(), estimate.loss);

    auto double_channel = sender.MakeDoubleChannel("Double");
    auto bool_channel = sender.MakeBoolChannel("Bool");
    auto string_channel = sender.MakeStringChannel("String");
    auto double_subscriber = table->GetDoubleTopic("Double").Subscribe(-1.0);
    auto bool_subscriber = table->GetBooleanTopic("Bool").Subscribe(false);
    auto string_subscriber = table->GetStringTopic("String").Subscribe("");

    double_channel(1.25);
    bool_channel(true);
    string_channel("ok");
    instance.FlushLocal();

    EXPECT_DOUBLE_EQ(double_subscriber.Get(), 1.25);
    EXPECT_TRUE(bool_subscriber.Get());
    EXPECT_EQ(string_subscriber.Get(), "ok");
  }

  instance.StopLocal();
  wpi::nt::NetworkTableInstance::Destroy(instance);
}

}  // namespace
