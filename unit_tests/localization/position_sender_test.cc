#include "gtest/gtest.h"
#include "localization/position_sender.h"
#include "localization/sim_sender.h"

namespace {

TEST(SimSenderTest, StoresLastSend) {
  localization::SimSender sender;
  auto ctx = std::make_shared<control_loops::Context>();
  localization::position_estimate_t estimate{
      .tag_ids = {1, 2},
      .pose = wpi::math::Pose3d{wpi::units::meter_t{1.0},
                                wpi::units::meter_t{2.0},
                                wpi::units::meter_t{3.0},
                                wpi::math::Rotation3d{}},
      .variance = 4.0,
      .num_tags = 2,
      .avg_tag_dist = 5.0};

  sender.Send(estimate, {{.camera_idx = 3, .timestamp = 77}}, ctx);

  ASSERT_TRUE(sender.last_estimate.has_value());
  EXPECT_EQ(sender.last_estimate->tag_ids, (std::vector<int>{1, 2}));
  ASSERT_EQ(sender.last_metadata.size(), 1);
  EXPECT_EQ(sender.last_metadata.front().camera_idx, 3);
  EXPECT_EQ(sender.last_metadata.front().timestamp, 77UL);
  EXPECT_EQ(sender.last_context, ctx);
}

TEST(PositionSenderInterfaceTest, SimSenderImplementsInterface) {
  EXPECT_TRUE(
      (std::is_base_of_v<localization::IPositionSender,
                         localization::SimSender>));
}

}  // namespace
