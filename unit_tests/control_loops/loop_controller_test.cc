#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "control_loops/loop_controller.h"
#include "gtest/gtest.h"

namespace {

TEST(LoopControllerTest, DispatchesLatestFrameAndMetadata) {
  auto controller = std::make_shared<control_loops::LoopController>(1);
  auto frame = std::make_shared<camera::JpegBuffer>(3);
  std::atomic<int> callback_count{0};

  controller->RegisterIterationCallback(
      0, [&](std::shared_ptr<camera::JpegBuffer> observed_frame,
             control_loops::MetaDataList metadata,
             std::shared_ptr<control_loops::Context> ctx) {
        EXPECT_EQ(observed_frame, frame);
        ASSERT_EQ(metadata.size(), 1);
        EXPECT_EQ(metadata.front().camera_idx, 0);
        EXPECT_EQ(metadata.front().timestamp, 99UL);
        EXPECT_NE(ctx, nullptr);
        callback_count.fetch_add(1);
        controller->RequestStop();
      });

  controller->ReceiveFrame(0, frame, 99);
  std::jthread run_thread([controller] { controller->Run(); });

  for (int i = 0; i < 50 && callback_count.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  controller->RequestStop();

  EXPECT_EQ(callback_count.load(), 1);
}

TEST(ContextTest, DestructionWakesControllerRunLoop) {
  auto controller = std::make_shared<control_loops::LoopController>(1);
  std::atomic<int> callback_count{0};

  controller->RegisterIterationCallback(
      0, [&](std::shared_ptr<camera::JpegBuffer>,
             control_loops::MetaDataList,
             std::shared_ptr<control_loops::Context> ctx) {
        callback_count.fetch_add(1);
        controller->RequestStop();
      });

  std::jthread run_thread([controller] { controller->Run(); });

  for (int i = 0; i < 50 && callback_count.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  controller->RequestStop();

  EXPECT_EQ(callback_count.load(), 1);
}

}  // namespace
