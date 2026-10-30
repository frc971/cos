#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <stop_token>

#include <gtest/gtest.h>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/ball_pathing_node.h"

namespace {

using namespace std::chrono_literals;

constexpr std::string_view kInputChannel = "ball_pathing/image";
constexpr std::string_view kOutputChannel = "ball_pathing/velocity_goal";

class BallPathingNodeTest
    : public testing::TestWithParam<gamepiece::UnetInputMode> {
 protected:
  void SetUp() override {
    const bool rgb = GetParam() == gamepiece::UnetInputMode::kRgb;
    const char* configured_engine = std::getenv(
        rgb ? "COS_UNET_TEST_RGB_ENGINE" : "COS_UNET_TEST_ENGINE");
    engine_ = configured_engine != nullptr
        ? configured_engine
        : rgb ? "/root/models/ball_unet_rgb.engine"
              : "/root/gamepiece_models/ball_unet.engine";
    if (!std::filesystem::is_regular_file(engine_)) {
      GTEST_SKIP() << "Requires a TensorRT engine: " << engine_;
    }
    const char* configured_constants = std::getenv("COS_TEST_CONSTANTS_DIR");
    constants_ = configured_constants != nullptr
        ? configured_constants
        : COS_TEST_CONSTANTS_DIR;
    node_ = std::make_unique<gamepiece::BallPathingNode>(
        kInputChannel, kOutputChannel, engine_,
        (constants_ / "vision-bot/front.json").string(), GetParam(), thread_pool_);
  }

  void TearDown() override { thread_pool_.Shutdown(); }

  static auto MakeContext() -> control_loop::Context {
    return std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
  }

  auto MakeBlackFrame()
      -> std::shared_ptr<camera::DecodedJpegBuffer> {
    auto frame = std::make_shared<camera::DecodedJpegBuffer>();
    frame->width = 1280;
    frame->height = 800;
    frame->output_format = GetParam() == gamepiece::UnetInputMode::kRgb
        ? NVJPEG_OUTPUT_RGBI : NVJPEG_OUTPUT_Y;
    frame->stride = frame->width * static_cast<int>(GetParam()) + 32;
    frame->channel_sizes[0] = frame->stride * frame->height;
    frame->output_size = frame->channel_sizes[0];
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(
                             &frame->destination.channel[0]),
                         frame->output_size),
              cudaSuccess);
    EXPECT_EQ(cudaMemset(frame->destination.channel[0], 0, frame->output_size),
              cudaSuccess);
    return frame;
  }

  std::string engine_;
  std::filesystem::path constants_;
  control_loop::ThreadPool thread_pool_{1};
  std::unique_ptr<gamepiece::BallPathingNode> node_;
};

TEST_P(BallPathingNodeTest, DeclaresImageInputAndVelocityGoalOutput) {
  ASSERT_EQ(node_->GetDependencies().size(), 1U);
  EXPECT_EQ(node_->GetDependencies()[0].GetChannel(), kInputChannel);
  EXPECT_TRUE(node_->GetDependencies()[0].GetTypes().contains(
      typeid(camera::DecodedJpegBuffer)));
  ASSERT_EQ(node_->GetPublications().size(), 1U);
  EXPECT_EQ(node_->GetPublications()[0].GetChannel(), kOutputChannel);
  EXPECT_TRUE(node_->GetPublications()[0].GetTypes().contains(
      typeid(gamepiece::BallVelocityGoalMessage)));
}

TEST_P(BallPathingNodeTest, MissingFramePublishesInvalidZeroGoal) {
  const auto context = MakeContext();
  size_t callbacks = 0;
  node_->RegisterCallback([&](const control_loop::Context& published) {
    const auto* goal =
        published->GetMessage<gamepiece::BallVelocityGoalMessage>(
            kOutputChannel);
    ASSERT_NE(goal, nullptr);
    EXPECT_FALSE(goal->valid);
    EXPECT_DOUBLE_EQ(goal->vx_mps, 0.0);
    EXPECT_DOUBLE_EQ(goal->vy_mps, 0.0);
    ++callbacks;
  });
  node_->CreateCallback()(context);
  EXPECT_EQ(callbacks, 1U);
}

TEST_P(BallPathingNodeTest, ProcessesGpuFrameThroughTensorRt) {
  const auto context = MakeContext();
  context->SetMessage(kInputChannel, MakeBlackFrame());
  std::promise<gamepiece::BallVelocityGoalMessage> published;
  auto result = published.get_future();
  node_->RegisterCallback([&](const control_loop::Context& output) {
    const auto* goal =
        output->GetMessage<gamepiece::BallVelocityGoalMessage>(kOutputChannel);
    ASSERT_NE(goal, nullptr);
    published.set_value(*goal);
  });
  node_->CreateCallback()(context);
  ASSERT_EQ(result.wait_for(5s), std::future_status::ready);
  const auto goal = result.get();
  EXPECT_TRUE(std::isfinite(goal.vx_mps));
  EXPECT_TRUE(std::isfinite(goal.vy_mps));
  EXPECT_TRUE(std::isfinite(goal.target_x_m));
  EXPECT_TRUE(std::isfinite(goal.target_y_m));
}

TEST_P(BallPathingNodeTest, RejectsEngineWithDifferentInputMode) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto other_mode = GetParam() == gamepiece::UnetInputMode::kRgb
      ? gamepiece::UnetInputMode::kGrayscale : gamepiece::UnetInputMode::kRgb;
  EXPECT_DEATH({
    gamepiece::BallPathingNode mismatched(
        kInputChannel, kOutputChannel, engine_,
        (constants_ / "vision-bot/front.json").string(), other_mode, thread_pool_);
  }, "Engine channels do not match");
}

INSTANTIATE_TEST_SUITE_P(
    InputModes, BallPathingNodeTest,
    testing::Values(gamepiece::UnetInputMode::kGrayscale,
                    gamepiece::UnetInputMode::kRgb));

}  // namespace
