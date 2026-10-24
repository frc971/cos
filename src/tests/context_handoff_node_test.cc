#include "control_loop/context_handoff_node.h"
#include "control_loop/control_loop.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

class SourceNode final : public control_loop::INode {
 public:
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };
  }
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) override {
    callbacks_.push_back(callback);
  }

 private:
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_{
      {"frame", typeid(control_loop::ValueMessage<int>)}};
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

auto MakeContext(std::uint64_t id) -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, id);
}

TEST(ContextHandoffNodeTest, PublishesNullForAnEmptySourceNotification) {
  auto source = std::make_shared<SourceNode>();
  control_loop::ContextHandoffNode handoff(source);
  auto input = MakeContext(1);
  input->SetMessage("frame", nullptr);
  source->CreateCallback()(input);
  auto output = MakeContext(2);
  handoff.CreateCallback()(output);
  EXPECT_TRUE(output->Exists("frame"));
  EXPECT_EQ(output->GetMessage<control_loop::ValueMessage<int>>("frame"), nullptr);
}

TEST(ContextHandoffNodeTest, IdleSourceDoesNotBlockReceivingLoopShutdown) {
  auto source = std::make_shared<SourceNode>();
  control_loop::ContextHandoffNode handoff(source);
  auto output = MakeContext(1);
  handoff.CreateCallback()(output);
  EXPECT_TRUE(output->Exists("frame"));
  EXPECT_EQ(output->GetMessage<control_loop::ValueMessage<int>>("frame"), nullptr);
}

TEST(ContextHandoffNodeTest, SlowReceiverDoesNotHoldUpCameraLoop) {
  using namespace std::chrono_literals;
  auto source = std::make_shared<SourceNode>();
  auto handoff = std::make_shared<control_loop::ContextHandoffNode>(source);
  std::atomic<int> captures = 0;
  std::atomic<bool> entered = false;
  std::promise<void> release; auto ready = release.get_future().share();
  source->RegisterCallback([&](const auto&) { ++captures; });
  handoff->RegisterCallback([&](const auto&) { if (!entered.exchange(true)) ready.wait(); });
  control_loop::ControlLoop camera_loop(1ms), worker_loop(1ms);
  camera_loop.RegisterDependencyNode(source); worker_loop.RegisterDependencyNode(handoff);
  camera_loop.Start(); worker_loop.Start();
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (!entered && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
  const int before = captures; std::this_thread::sleep_for(20ms);
  EXPECT_TRUE(entered); EXPECT_GT(captures, before);
  release.set_value(); camera_loop.Stop(); worker_loop.Stop();
}

TEST(ContextHandoffNodeTest, RetainsLatestFrameAndConsumesItOnce) {
  auto source = std::make_shared<SourceNode>();
  control_loop::ContextHandoffNode handoff(source);
  auto input = MakeContext(1);
  input->SetMessage("frame",
                    std::make_unique<control_loop::ValueMessage<int>>(7));
  source->CreateCallback()(input);
  auto empty = MakeContext(2);
  empty->SetMessage("frame", nullptr);
  source->CreateCallback()(empty);
  auto output = MakeContext(3);
  handoff.CreateCallback()(output);
  const auto* frame = output->GetMessage<control_loop::ValueMessage<int>>("frame");
  ASSERT_NE(frame, nullptr);
  EXPECT_EQ(frame->value, 7);
  source->CreateCallback()(empty);
  auto consumed = MakeContext(4);
  handoff.CreateCallback()(consumed);
  EXPECT_TRUE(consumed->Exists("frame"));
  EXPECT_EQ(consumed->GetMessage<control_loop::ValueMessage<int>>("frame"), nullptr);
}

}  // namespace
