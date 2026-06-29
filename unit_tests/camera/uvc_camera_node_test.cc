#include <cstring>
#include <filesystem>
#include <type_traits>

#include "camera/uvc_camera_node.h"
#include "gtest/gtest.h"
#include "unit_tests/test_helpers.h"

namespace {

TEST(UVCCameraConfigTest, BuildsFromCameraConstantWithDefaults) {
  camera::camera_constant_t constant;
  constant.name = "front";
  constant.serial_id = "serial";
  constant.frame_width = 640;
  constant.frame_height = 480;
  constant.fps = 30;

  const camera::UVCCameraConfig config(constant);

  EXPECT_EQ(config.name, "front");
  EXPECT_EQ(config.serial_id, "serial");
  EXPECT_EQ(config.width, 640);
  EXPECT_EQ(config.height, 480);
  EXPECT_EQ(config.fps, 30);
  EXPECT_EQ(config.max_payload_size, 3072);
  EXPECT_EQ(config.max_frame_size, 2048589);
}

TEST(UVCCameraConfigTest, BuildsFromJsonFile) {
  const auto path = cos_test::testing::WriteJsonFile(
      std::filesystem::temp_directory_path() / "uvc_camera_config_test.json",
      {{"camera_type", "uvc"}, {"name", "front"}, {"serial_id", "serial"},
       {"height", 480},        {"width", 640},    {"fps", 30},
       {"max_payload_size", 1}, {"max_frame_size", 2}});

  const camera::UVCCameraConfig config(path);

  EXPECT_EQ(config.name, "front");
  EXPECT_EQ(config.serial_id, "serial");
  EXPECT_EQ(config.width, 640);
  EXPECT_EQ(config.height, 480);
  EXPECT_EQ(config.fps, 30);
  EXPECT_EQ(config.max_payload_size, 1);
  EXPECT_EQ(config.max_frame_size, 2);
}

TEST(JpegBufferTest, OwnsWritableMemoryAndReportsSize) {
  camera::JpegBuffer buffer(4);
  ASSERT_NE(buffer.ptr(), nullptr);

  std::memcpy(buffer.ptr(), "test", 4);

  EXPECT_EQ(buffer.size(), 4U);
  EXPECT_EQ(std::memcmp(buffer.ptr(), "test", 4), 0);
}

TEST(UVCCameraNodeTest, IsMoveAndCopyDisabledByOwnedNativeHandles) {
  EXPECT_FALSE(std::is_copy_constructible_v<camera::UVCCameraNode>);
  EXPECT_FALSE(std::is_move_constructible_v<camera::UVCCameraNode>);
}

}  // namespace
