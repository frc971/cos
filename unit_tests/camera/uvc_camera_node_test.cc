#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "camera/disk_camera.h"
#include "camera/simulated_disk_camera.h"
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
      {{"camera_type", "uvc"},
       {"name", "front"},
       {"serial_id", "serial"},
       {"height", 480},
       {"width", 640},
       {"fps", 30},
       {"max_payload_size", 1},
       {"max_frame_size", 2}});

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
  EXPECT_TRUE((std::is_base_of_v<camera::ICamera, camera::UVCCameraNode>));
  EXPECT_FALSE(std::is_copy_constructible_v<camera::UVCCameraNode>);
  EXPECT_FALSE(std::is_move_constructible_v<camera::UVCCameraNode>);
}

TEST(DiskCameraTest, ImplementsCameraInterfaceAndPublishesDiskFrame) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "disk_camera_test.mjpg";
  {
    std::ofstream out(path, std::ios::binary);
    out << "jpeg";
  }

  camera::DiskCamera disk_camera(path, 1234);
  EXPECT_TRUE((std::is_base_of_v<camera::ICamera, camera::DiskCamera>));

  std::shared_ptr<camera::JpegBuffer> observed_frame;
  unsigned long observed_timestamp = 0;
  disk_camera.RegisterCallback(
      [&](std::shared_ptr<camera::JpegBuffer> frame, unsigned long timestamp) {
        observed_frame = std::move(frame);
        observed_timestamp = timestamp;
      });

  disk_camera.Start();

  ASSERT_NE(observed_frame, nullptr);
  EXPECT_EQ(observed_frame->size(), 4U);
  EXPECT_EQ(std::memcmp(observed_frame->ptr(), "jpeg", 4), 0);
  EXPECT_EQ(observed_timestamp, 1234UL);
}

TEST(SimulatedDiskCameraTest, PublishesFramesAtSimulatedTimestamps) {
  std::vector<std::vector<unsigned char>> frames = {{1}, {2}, {3}};
  camera::SimulatedDiskCamera camera(frames, 1000,
                                     std::chrono::microseconds(250),
                                     std::chrono::microseconds(1));
  EXPECT_TRUE(
      (std::is_base_of_v<camera::ICamera, camera::SimulatedDiskCamera>));

  std::mutex mutex;
  std::condition_variable received;
  std::vector<unsigned long> timestamps;
  std::vector<unsigned char> values;
  camera.RegisterCallback(
      [&](const std::shared_ptr<camera::JpegBuffer>& frame,
          unsigned long timestamp) {
        std::lock_guard<std::mutex> lock(mutex);
        timestamps.push_back(timestamp);
        values.push_back(static_cast<unsigned char*>(frame->ptr())[0]);
        received.notify_one();
      });

  camera.Start();

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(received.wait_for(lock, std::chrono::seconds(1),
                                [&] { return timestamps.size() == 3; }));
  EXPECT_EQ(timestamps, (std::vector<unsigned long>{1000, 1250, 1500}));
  EXPECT_EQ(values, (std::vector<unsigned char>{1, 2, 3}));
}


}  // namespace
