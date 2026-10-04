#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/control_loop.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/gamepiece_control_loop.h"

using namespace std::chrono_literals;

namespace {

class TestMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(TestMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
};

struct TestContext {
  TestContext()
      : context(new control_loop::ContextInternal(
            std::chrono::steady_clock::now(), nullptr, stop_source.get_token(),
            0)),
        weak_context(context) {}

  std::stop_source stop_source;
  control_loop::Context context;
  std::weak_ptr<control_loop::ContextInternal> weak_context;
};

class FakeDecoderNode final : public control_loop::INode {
 public:
  explicit FakeDecoderNode(std::string_view channel)
      : publications_({{channel, typeid(camera::DecodedJpegBuffer)}}) {}

  void Emit(const control_loop::Context& context) {
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  }

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [](const control_loop::Context&) {};
  }

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override {
    callbacks_.push_back(callback);
  }

  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }

  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }

 private:
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

class FakeGamepieceNode final : public control_loop::INode {
 public:
  using Observer =
      std::function<void(const control_loop::Context&,
                         const std::shared_ptr<camera::DecodedJpegBuffer>&)>;

  FakeGamepieceNode(std::string_view channel, Observer observer)
      : channel_(channel),
        observer_(std::move(observer)),
        dependencies_({{channel_, typeid(camera::DecodedJpegBuffer)}}) {}

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      auto frame =
          context->GetSharedMessage<camera::DecodedJpegBuffer>(channel_);
      if (frame != nullptr) {
        observer_(context, frame);
      }
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };
  }

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override {
    callbacks_.push_back(callback);
  }

  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }

  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }

 private:
  std::string channel_;
  Observer observer_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

auto EmitFrame(FakeDecoderNode& decoder, std::string_view channel,
               double timestamp)
    -> std::weak_ptr<camera::DecodedJpegBuffer> {
  TestContext localization;
  auto frame = std::make_shared<camera::DecodedJpegBuffer>();
  frame->timestamp = timestamp;
  std::weak_ptr<camera::DecodedJpegBuffer> weak_frame = frame;
  localization.context->SetMessage(channel, frame);
  localization.context->SetMessage("localization-only",
                                   std::make_unique<TestMessage>());
  decoder.Emit(localization.context);
  frame.reset();
  localization.context.reset();
  EXPECT_TRUE(localization.weak_context.expired());
  return weak_frame;
}

TEST(ContextSharedMessageTest, SharedMessageOutlivesContext) {
  std::shared_ptr<TestMessage> retained;
  std::weak_ptr<TestMessage> weak_message;
  TestContext owner;
  {
    auto message = std::make_shared<TestMessage>();
    weak_message = message;
    owner.context->SetMessage("shared", message);
    retained = owner.context->GetSharedMessage<TestMessage>("shared");
    ASSERT_EQ(retained, message);
  }

  owner.context.reset();
  EXPECT_TRUE(owner.weak_context.expired());
  EXPECT_FALSE(weak_message.expired());
  retained.reset();
  EXPECT_TRUE(weak_message.expired());
}

TEST(GamepieceControlLoopTest, KeepsInFlightFrameAndConsumesLatestFrame) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<double> observed_timestamps;
  bool first_frame_entered = false;
  bool release_first_frame = false;
  bool copied_localization_message = false;

  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::unique_lock lock(mutex);
        copied_localization_message |=
            context->GetMessage<TestMessage>("localization-only") != nullptr;
        if (observed_timestamps.size() < 32U) {
          observed_timestamps.push_back(frame->timestamp);
        }
        if (frame->timestamp == 1.0 && !first_frame_entered) {
          first_frame_entered = true;
          condition.notify_all();
          condition.wait(lock, [&] { return release_first_frame; });
        }
        condition.notify_all();
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);

  std::weak_ptr<camera::DecodedJpegBuffer> first_frame =
      EmitFrame(*decoder, kChannel, 1.0);
  loop.Start();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return first_frame_entered;
    }));
  }

  const std::weak_ptr<camera::DecodedJpegBuffer> second_frame =
      EmitFrame(*decoder, kChannel, 2.0);
  const std::weak_ptr<camera::DecodedJpegBuffer> third_frame =
      EmitFrame(*decoder, kChannel, 3.0);
  EXPECT_FALSE(first_frame.expired());
  EXPECT_TRUE(second_frame.expired());
  EXPECT_FALSE(third_frame.expired());

  {
    std::lock_guard lock(mutex);
    release_first_frame = true;
  }
  condition.notify_all();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return std::ranges::find(observed_timestamps, 3.0) !=
             observed_timestamps.end();
    }));
  }
  loop.Stop();

  EXPECT_TRUE(first_frame.expired());
  EXPECT_TRUE(third_frame.expired());
  EXPECT_FALSE(copied_localization_message);
  EXPECT_EQ(std::ranges::find(observed_timestamps, 2.0),
            observed_timestamps.end());
}

TEST(GamepieceControlLoopTest, RunsNodesSequentiallyWhenOneOverruns) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<double> started_frames;
  bool release_first_frame = false;
  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        (void)context;
        std::unique_lock lock(mutex);
        started_frames.push_back(frame->timestamp);
        condition.notify_all();
        if (frame->timestamp == 1.0) {
          condition.wait(lock, [&] { return release_first_frame; });
        }
      });

  gamepiece::GamepieceControlLoop loop(10ms);
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);
  EmitFrame(*decoder, kChannel, 1.0);
  loop.Start();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return started_frames.size() == 1;
    }));
  }

  EmitFrame(*decoder, kChannel, 2.0);
  {
    std::unique_lock lock(mutex);
    EXPECT_FALSE(condition.wait_for(lock, 30ms, [&] {
      return started_frames.size() == 2;
    }));
    release_first_frame = true;
  }
  condition.notify_all();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return started_frames.size() == 2;
    }));
    ASSERT_EQ(started_frames.size(), 2U);
    EXPECT_EQ(started_frames[0], 1.0);
    EXPECT_EQ(started_frames[1], 2.0);
  }
  loop.Stop();
}

TEST(GamepieceControlLoopTest, WaitsForAsyncWorkBeforeTakingLatestFrame) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);
  control_loop::ThreadPool thread_pool(1);
  std::mutex mutex;
  std::condition_variable condition;
  std::vector<double> submitted_frames;
  bool first_frame_entered = false;
  bool release_first_frame = false;
  bool third_frame_completed = false;

  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        {
          std::lock_guard lock(mutex);
          submitted_frames.push_back(frame->timestamp);
        }
        thread_pool.Submit([&, context, frame] {
          std::unique_lock lock(mutex);
          if (frame->timestamp == 1.0) {
            first_frame_entered = true;
            condition.notify_all();
            condition.wait(lock, [&] { return release_first_frame; });
          }
          third_frame_completed |= frame->timestamp == 3.0;
          condition.notify_all();
        }, context->id);
      });

  gamepiece::GamepieceControlLoop loop(10ms);
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);
  const auto first_frame = EmitFrame(*decoder, kChannel, 1.0);
  loop.Start();
  bool entered;
  {
    std::unique_lock lock(mutex);
    entered = condition.wait_for(lock, 2s, [&] {
      return first_frame_entered;
    });
  }
  const auto second_frame = EmitFrame(*decoder, kChannel, 2.0);
  std::this_thread::sleep_for(30ms);
  const auto third_frame = EmitFrame(*decoder, kChannel, 3.0);
  std::this_thread::sleep_for(30ms);
  const bool first_frame_retained = !first_frame.expired();
  const bool second_frame_dropped = second_frame.expired();
  size_t submissions_while_blocked;
  {
    std::lock_guard lock(mutex);
    submissions_while_blocked = submitted_frames.size();
    release_first_frame = true;
  }
  condition.notify_all();
  bool completed;
  {
    std::unique_lock lock(mutex);
    completed = condition.wait_for(lock, 2s, [&] {
      return third_frame_completed;
    });
  }
  loop.Stop();
  thread_pool.Shutdown();

  EXPECT_TRUE(entered);
  EXPECT_TRUE(completed);
  EXPECT_TRUE(first_frame_retained);
  EXPECT_TRUE(second_frame_dropped);
  EXPECT_EQ(submissions_while_blocked, 1U);
  EXPECT_EQ(submitted_frames, (std::vector<double>{1.0, 3.0}));
  EXPECT_TRUE(first_frame.expired());
  EXPECT_TRUE(third_frame.expired());
}

TEST(GamepieceControlLoopTest, SharesWorkersWithIndependentLocalizationLoop) {
  constexpr std::string_view kChannel = "decoded/front";
  control_loop::ThreadPool thread_pool(2);
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);
  std::mutex mutex;
  std::condition_variable condition;
  bool gamepiece_entered = false;
  bool release_gamepiece = false;
  size_t localization_completions = 0;
  std::thread::id gamepiece_scheduler;
  std::thread::id gamepiece_worker;
  std::thread::id localization_scheduler;
  std::thread::id localization_worker;

  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>&) {
        {
          std::lock_guard lock(mutex);
          gamepiece_scheduler = std::this_thread::get_id();
        }
        thread_pool.Submit([&, context] {
          std::unique_lock lock(mutex);
          gamepiece_worker = std::this_thread::get_id();
          gamepiece_entered = true;
          condition.notify_all();
          condition.wait(lock, [&] { return release_gamepiece; });
        }, context->id);
      });

  control_loop::ControlLoop localization_loop(1ms);
  localization_loop.RegisterCallback(
      [&](const control_loop::Context& context) {
        {
          std::lock_guard lock(mutex);
          localization_scheduler = std::this_thread::get_id();
        }
        thread_pool.Submit([&, context] {
          std::lock_guard lock(mutex);
          localization_worker = std::this_thread::get_id();
          ++localization_completions;
          condition.notify_all();
        }, context->id);
      });
  gamepiece::GamepieceControlLoop gamepiece_loop(10ms);
  gamepiece_loop.RegisterDecodedFrameSource(decoder, kChannel);
  gamepiece_loop.RegisterNode(consumer);
  EmitFrame(*decoder, kChannel, 1.0);
  gamepiece_loop.Start();
  bool entered;
  {
    std::unique_lock lock(mutex);
    entered = condition.wait_for(lock, 2s, [&] { return gamepiece_entered; });
  }

  localization_loop.Start();
  bool localization_progressed;
  {
    std::unique_lock lock(mutex);
    localization_progressed = condition.wait_for(
        lock, 2s, [&] { return localization_completions >= 3; });
    release_gamepiece = true;
  }
  condition.notify_all();
  localization_loop.Stop();
  gamepiece_loop.Stop();
  thread_pool.Shutdown();

  EXPECT_TRUE(entered);
  EXPECT_TRUE(localization_progressed);
  EXPECT_NE(gamepiece_scheduler, localization_scheduler);
  EXPECT_NE(gamepiece_scheduler, gamepiece_worker);
  EXPECT_NE(localization_scheduler, localization_worker);
}

TEST(GamepieceControlLoopTest, KeepsCameraChannelsIndependent) {
  constexpr std::string_view kFrontChannel = "decoded/front";
  constexpr std::string_view kRearChannel = "decoded/rear";
  auto front_decoder = std::make_shared<FakeDecoderNode>(kFrontChannel);
  auto rear_decoder = std::make_shared<FakeDecoderNode>(kRearChannel);

  std::mutex mutex;
  std::condition_variable condition;
  bool saw_front = false;
  bool saw_rear = false;
  bool mixed_channels = false;

  auto front_consumer = std::make_shared<FakeGamepieceNode>(
      kFrontChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::lock_guard lock(mutex);
        saw_front |= frame->timestamp == 11.0;
        mixed_channels |=
            context
                ->GetSharedMessage<camera::DecodedJpegBuffer>(kRearChannel) !=
            nullptr;
        condition.notify_all();
      });
  auto rear_consumer = std::make_shared<FakeGamepieceNode>(
      kRearChannel,
      [&](const control_loop::Context&,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::lock_guard lock(mutex);
        saw_rear |= frame->timestamp == 22.0;
        condition.notify_all();
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(front_decoder, kFrontChannel);
  loop.RegisterDecodedFrameSource(rear_decoder, kRearChannel);
  loop.RegisterNode(front_consumer);
  loop.RegisterNode(rear_consumer);
  EmitFrame(*front_decoder, kFrontChannel, 11.0);
  EmitFrame(*rear_decoder, kRearChannel, 22.0);

  loop.Start();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s,
                                   [&] { return saw_front && saw_rear; }));
  }
  loop.Stop();

  EXPECT_FALSE(mixed_channels);
}

TEST(GamepieceControlLoopTest, StopsBeforeAnyFrameArrives) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);
  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [](const control_loop::Context&,
         const std::shared_ptr<camera::DecodedJpegBuffer>&) {
        FAIL() << "A gamepiece node ran without a decoded frame";
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);
  loop.Start();
  std::this_thread::sleep_for(10ms);

  const auto stop_start = std::chrono::steady_clock::now();
  loop.Stop();
  EXPECT_LT(std::chrono::steady_clock::now() - stop_start, 500ms);
}

}  // namespace
