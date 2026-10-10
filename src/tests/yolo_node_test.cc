#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/yolo_node.h"

using namespace std::chrono_literals;

namespace {

constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
constexpr std::string_view kDetectionsChannel = "gamepiece/detections";

class YoloNodeTest : public testing::Test {
 protected:
  void SetUp() override {
    const char* configured_model = std::getenv("COS_YOLO_TEST_ENGINE");
    const std::string model = configured_model != nullptr
        ? configured_model : "/root/gamepiece_models/best_nms_gray.engine";
    if (!std::filesystem::is_regular_file(model)) {
      GTEST_SKIP() << "Requires a TensorRT engine: " << model;
    }
    node_ = std::make_unique<gamepiece::YoloNode>(
        kDecodedChannel, kDetectionsChannel, model,
        std::vector<std::string>{}, thread_pool_);
  }

  void TearDown() override { thread_pool_.Shutdown(); }

  static auto MakeContext(std::uint64_t id = 0) -> control_loop::Context {
    return std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, id);
  }

  static auto MakeFrame() -> std::shared_ptr<camera::DecodedJpegBuffer> {
    auto frame = std::make_shared<camera::DecodedJpegBuffer>();
    frame->width = 32;
    frame->height = 16;
    frame->stride = 64;
    frame->channel_sizes[0] = frame->stride * frame->height;
    frame->output_size = frame->channel_sizes[0];
    EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(
                             &frame->destination.channel[0]),
                         frame->output_size), cudaSuccess);
    EXPECT_EQ(cudaMemset(frame->destination.channel[0], 0, frame->output_size),
              cudaSuccess);
    return frame;
  }

  control_loop::ThreadPool thread_pool_{2};
  std::unique_ptr<gamepiece::YoloNode> node_;
};

TEST_F(YoloNodeTest, DeclaresDecodedInputAndBoundingBoxVectorOutput) {
  ASSERT_EQ(node_->GetDependencies().size(), 1U);
  EXPECT_EQ(node_->GetDependencies()[0].GetChannel(), kDecodedChannel);
  EXPECT_TRUE(node_->GetDependencies()[0].GetTypes().contains(
      typeid(camera::DecodedJpegBuffer)));
  ASSERT_EQ(node_->GetPublications().size(), 1U);
  EXPECT_EQ(node_->GetPublications()[0].GetChannel(), kDetectionsChannel);
  EXPECT_TRUE(node_->GetPublications()[0].GetTypes().contains(
      typeid(control_loop::ValueMessage<
             gamepiece::bounding_box_detections_t>)));
}

TEST_F(YoloNodeTest, PublishesNullAndNotifiesForMissingOrInvalidFrames) {
  size_t callbacks = 0;
  node_->RegisterCallback([&](const control_loop::Context& context) -> void {
    EXPECT_TRUE(context->Exists(std::string(kDetectionsChannel)));
    EXPECT_EQ(context->GetMessage<
                  control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
                  kDetectionsChannel), nullptr);
    ++callbacks;
  });
  const auto run = node_->CreateCallback();
  run(MakeContext());
  auto null_input = MakeContext();
  null_input->SetMessage(kDecodedChannel, nullptr);
  run(null_input);
  auto empty_frame = MakeContext();
  empty_frame->SetMessage(kDecodedChannel,
                         std::make_shared<camera::DecodedJpegBuffer>());
  run(empty_frame);
  auto invalid_dimensions = MakeContext();
  auto frame = MakeFrame();
  frame->width = 0;
  invalid_dimensions->SetMessage(kDecodedChannel, std::move(frame));
  run(invalid_dimensions);
  EXPECT_EQ(callbacks, 4U);
}

TEST_F(YoloNodeTest, QueuesInferenceAndPublishesFromWorker) {
  // Occupy both workers so the callback must return before inference can run.
  std::promise<void> release_workers;
  const auto released = release_workers.get_future().share();
  std::promise<void> started_first;
  std::promise<void> started_second;
  auto first = started_first.get_future();
  auto second = started_second.get_future();
  thread_pool_.Submit([&]() -> void { started_first.set_value(); released.wait(); });
  thread_pool_.Submit([&]() -> void { started_second.set_value(); released.wait(); });
  EXPECT_EQ(first.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(second.wait_for(2s), std::future_status::ready);

  const auto context = MakeContext();
  context->SetMessage(kDecodedChannel, MakeFrame());
  std::promise<control_loop::Context> published;
  auto publication = published.get_future();
  node_->RegisterCallback([&](const control_loop::Context& published_context) -> void {
    published.set_value(published_context);
  });
  std::promise<void> submitted;
  auto submission = submitted.get_future();
  std::jthread submitter([&]() -> void {
    node_->CreateCallback()(context);
    submitted.set_value();
  });
  EXPECT_EQ(submission.wait_for(500ms), std::future_status::ready);
  EXPECT_EQ(publication.wait_for(50ms), std::future_status::timeout);
  EXPECT_FALSE(context->Exists(std::string(kDetectionsChannel)));
  release_workers.set_value();
  submitter.join();
  thread_pool_.Shutdown();

  ASSERT_EQ(publication.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(publication.get(), context);
  ASSERT_NE(context->GetMessage<
                control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
                kDetectionsChannel), nullptr);
}

TEST_F(YoloNodeTest, SerializesInferenceAcrossConcurrentContexts) {
  constexpr size_t kFrames = 8;
  std::atomic<size_t> callbacks = 0;
  node_->RegisterCallback([&](const control_loop::Context&) -> void { ++callbacks; });
  std::vector<control_loop::Context> contexts;
  const auto run = node_->CreateCallback();
  for (size_t i = 0; i < kFrames; ++i) {
    auto context = MakeContext(i);
    context->SetMessage(kDecodedChannel, MakeFrame());
    run(context);
    contexts.push_back(std::move(context));
  }
  thread_pool_.Shutdown();
  EXPECT_EQ(callbacks.load(), kFrames);
  const auto* first = contexts.front()->GetMessage<
      control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
      kDetectionsChannel);
  ASSERT_NE(first, nullptr);
  for (const auto& context : contexts) {
    const auto* result = context->GetMessage<
        control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
        kDetectionsChannel);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->value.size(), first->value.size());
    for (size_t i = 0; i < result->value.size(); ++i) {
      EXPECT_EQ(result->value[i].bounds, first->value[i].bounds);
      EXPECT_EQ(result->value[i].class_id, first->value[i].class_id);
      EXPECT_EQ(result->value[i].label, first->value[i].label);
      EXPECT_FLOAT_EQ(result->value[i].confidence, first->value[i].confidence);
    }
  }
}

}  // namespace
