#include <filesystem>

#include "camera/camera_constants.h"
#include "gtest/gtest.h"
#include "unit_tests/test_helpers.h"

namespace {

TEST(CameraConstantsTest, ParsesValidCamerasAndSkipsInvalidEntries) {
  const auto path = cos_test::testing::WriteJsonFile(
      std::filesystem::temp_directory_path() / "camera_constants_test.json",
      {{"cameras",
        {{{"name", "front"},
          {"pipeline", "pipe"},
          {"intrinsics_path", "intrinsics.json"},
          {"extrinsics_path", "extrinsics.json"},
          {"backlight", 1.5},
          {"frame_width", 640},
          {"frame_height", 480},
          {"fps", 30},
          {"exposure", 10.0},
          {"brightness", 2.0},
          {"sharpness", 3.0},
          {"max_frame_size", 1234},
          {"max_payload_size", 5678},
          {"serial_id", "abc"},
          {"stream_ratio", 0.5},
          {"port", 1181},
          {"streamer_fps", 15},
          {"detector_type", "opencv_cpu"},
          {"camera_type", "uvc"}},
         nullptr,
         {{"name", "front"}, {"camera_type", "mipi"}},
         {{"name", "rear"},
          {"detector_type", "austin_gpu"},
          {"camera_type", "opencv"}}}}});

  const camera::camera_constants_t constants = camera::GetCameraConstants(path);

  ASSERT_EQ(constants.size(), 2);
  const auto& front = constants.at("front");
  EXPECT_EQ(front.name, "front");
  EXPECT_EQ(front.pipeline, "pipe");
  EXPECT_EQ(front.intrinsics_path, "intrinsics.json");
  EXPECT_EQ(front.extrinsics_path, "extrinsics.json");
  EXPECT_EQ(front.backlight, 1.5);
  EXPECT_EQ(front.frame_width, 640U);
  EXPECT_EQ(front.frame_height, 480U);
  EXPECT_EQ(front.fps, 30U);
  EXPECT_EQ(front.exposure, 10.0);
  EXPECT_EQ(front.brightness, 2.0);
  EXPECT_EQ(front.sharpness, 3.0);
  EXPECT_EQ(front.max_frame_size, 1234U);
  EXPECT_EQ(front.max_payload_size, 5678U);
  EXPECT_EQ(front.serial_id, "abc");
  EXPECT_EQ(front.stream_ratio, 0.5);
  EXPECT_EQ(front.port, 1181U);
  EXPECT_EQ(front.streamer_fps, 15U);
  EXPECT_EQ(front.detector_type, camera::DetectorType::OPENCV_CPU);
  EXPECT_EQ(front.camera_type, camera::CameraType::UVC);

  EXPECT_EQ(constants.at("rear").detector_type,
            camera::DetectorType::AUSTIN_GPU);
  EXPECT_EQ(constants.at("rear").camera_type, camera::CameraType::OPENCV);
}

TEST(CameraConstantTest, SortsByName) {
  camera::camera_constant_t a{.name = "a"};
  camera::camera_constant_t b{.name = "b"};

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
}

}  // namespace
