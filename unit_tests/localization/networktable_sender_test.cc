#include "gtest/gtest.h"
#include "localization/networktable_sender.h"

namespace {

TEST(NetworkTableSenderTest, SendAndAdHocChannelsDoNotThrow) {
  localization::NetworkTableSender sender("unit-test-camera");
  localization::position_estimate_t estimate{
      .tag_ids = {1, 49, 50},
      .rejected_tag_ids = {2, 99},
      .pose = wpi::math::Pose3d{wpi::units::meter_t{1.0},
                                wpi::units::meter_t{2.0},
                                wpi::units::meter_t{0.0},
                                wpi::math::Rotation3d{}},
      .variance = 0.75,
      .num_tags = 3,
      .avg_tag_dist = 2.5,
      .loss = 0.1};

  EXPECT_NO_THROW(sender.Send(estimate, {{.camera_idx = 0, .timestamp = 100},
                                         {.camera_idx = 1, .timestamp = 120}},
                              nullptr));

  auto double_channel = sender.MakeDoubleChannel("Double");
  auto bool_channel = sender.MakeBoolChannel("Bool");
  auto string_channel = sender.MakeStringChannel("String");

  EXPECT_NO_THROW(double_channel(1.25));
  EXPECT_NO_THROW(bool_channel(true));
  EXPECT_NO_THROW(string_channel("ok"));
}

}  // namespace
