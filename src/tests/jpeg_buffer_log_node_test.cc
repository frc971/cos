#include "logging/jpeg_buffer_log_node.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <networktables/BooleanTopic.h>
#include <networktables/NetworkTableInstance.h>
#include <wpi/timestamp.h>

#include "camera/jpeg_buffer.h"
#include "control_loop/connect_to_rio.h"

namespace {
using namespace std::chrono_literals;
constexpr std::string_view kEnabled = "/AdvantageKit/DriverStation/Enabled";

class JpegBufferLogNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    folder_ = std::filesystem::temp_directory_path() /
              ("cos-jpeg-log-test-" + std::to_string(wpi::Now()));
    std::filesystem::create_directories(folder_);
    client_.StopClient();
    EXPECT_FALSE(control_loop::ShouldLogImages());
    server_ = nt::NetworkTableInstance::Create();
    server_.StartServer("", "127.0.0.1", 0, 5811);
    client_.SetServer("127.0.0.1", 5811);
    client_.StartClient4("jpeg_buffer_log_node_test");
    ASSERT_TRUE(WaitFor([this]() -> bool { return client_.IsConnected(); }));
    EXPECT_FALSE(control_loop::ShouldLogImages());
    node_ = std::make_unique<logging::JpegBufferLogNode>(
        "jpeg", folder_.string(), pool_, control_loop::ShouldLogImages);
    node_->RegisterCallback([this](const control_loop::Context&) -> void {
      completed_.set_value();
    });
  }

  void TearDown() override {
    pool_.Shutdown();
    node_.reset();
    client_.StopClient();
    EXPECT_FALSE(control_loop::ShouldLogImages());
    server_.StopServer();
    nt::NetworkTableInstance::Destroy(server_);
    std::filesystem::remove_all(folder_);
  }

  template <typename Predicate>
  static auto WaitFor(Predicate predicate) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(5ms);
    }
    return true;
  }

  void SetEnabled(bool enabled) {
    ASSERT_TRUE(server_.GetEntry(kEnabled).SetBoolean(enabled));
    server_.Flush();
    ASSERT_TRUE(WaitFor([this, enabled]() -> bool {
      const auto value = client_.GetEntry(kEnabled).GetValue();
      return value.IsBoolean() && value.GetBoolean() == enabled;
    }));
  }

  auto QueueFrame() -> std::future<void> {
    completed_ = std::promise<void>{};
    auto done = completed_.get_future();
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, ++id_);
    auto jpeg = std::make_unique<camera::JpegBuffer>(4, id_);
    std::memcpy(jpeg->ptr, "jpeg", 4);
    context->SetMessage("jpeg", std::move(jpeg));
    node_->CreateCallback()(context);
    return done;
  }

  void Record(bool expected) {
    auto done = QueueFrame();
    ASSERT_EQ(done.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(std::filesystem::exists(FramePath()), expected);
    if (expected) {
      std::ifstream input(FramePath(), std::ios::binary);
      std::string bytes(4, '\0');
      input.read(bytes.data(), bytes.size());
      EXPECT_EQ(bytes, "jpeg");
    }
  }

  auto FramePath() const -> std::filesystem::path {
    return folder_ / (std::to_string(static_cast<double>(id_)) + ".jpg");
  }

  std::filesystem::path folder_;
  nt::NetworkTableInstance server_;
  nt::NetworkTableInstance client_ = nt::NetworkTableInstance::GetDefault();
  control_loop::ThreadPool pool_{1};
  std::unique_ptr<logging::JpegBufferLogNode> node_;
  std::promise<void> completed_;
  uint64_t id_ = 0;
};

TEST_F(JpegBufferLogNodeTest, RecordsFromFirstEnableThroughSecondDisable) {
  Record(false);  // No enabled topic yet.
  SetEnabled(false);
  Record(false);
  // Two matches exercise rearming after the final disable.
  for (int match = 0; match < 2; ++match) {
    SetEnabled(true);
    Record(true);
    Record(true);  // Repeated checks by multiple cameras must not count enables.
    SetEnabled(false);
    Record(true);
    Record(true);  // Keep logging for the entire auto-to-teleop gap.
    SetEnabled(true);
    Record(true);
    SetEnabled(false);
    Record(false);
    Record(false);
  }
}

TEST_F(JpegBufferLogNodeTest, ProcessesTransitionsBetweenFrames) {
  SetEnabled(false);
  Record(false);
  SetEnabled(true);
  SetEnabled(false);
  Record(true);  // Both auto transitions occurred between frames.
  SetEnabled(true);
  SetEnabled(false);
  Record(false);  // Both teleop transitions occurred between frames.
}

TEST_F(JpegBufferLogNodeTest, StopsAndResetsWhenDisconnected) {
  SetEnabled(true);
  Record(true);
  SetEnabled(false);
  Record(true);
  client_.StopClient();
  ASSERT_TRUE(WaitFor([this]() -> bool { return !client_.IsConnected(); }));
  Record(false);
  client_.StartClient4("jpeg_buffer_log_node_test");
  ASSERT_TRUE(WaitFor([this]() -> bool { return client_.IsConnected(); }));
  SetEnabled(false);
  Record(false);
  SetEnabled(true);
  Record(true);
  SetEnabled(false);
  Record(true);  // The reconnect starts a new cycle.
}

TEST_F(JpegBufferLogNodeTest, StopsWhenEnabledTopicIsUnpublished) {
  auto publisher = server_.GetBooleanTopic(kEnabled).Publish();
  publisher.Set(true);
  server_.Flush();
  ASSERT_TRUE(WaitFor([]() -> bool { return control_loop::ShouldLogImages(); }));
  Record(true);
  publisher = {};
  ASSERT_TRUE(WaitFor([]() -> bool { return !control_loop::ShouldLogImages(); }));
  Record(false);
}

TEST_F(JpegBufferLogNodeTest, KeepsQueuedFramesAcrossFirstDisable) {
  SetEnabled(true);
  ASSERT_TRUE(control_loop::ShouldLogImages());
  std::promise<void> started;
  std::promise<void> release;
  auto released = release.get_future();
  pool_.Submit([&]() -> void {
    started.set_value();
    released.wait();
  });
  started.get_future().wait();
  auto done = QueueFrame();
  SetEnabled(false);
  release.set_value();
  ASSERT_EQ(done.wait_for(3s), std::future_status::ready);
  EXPECT_TRUE(std::filesystem::exists(FramePath()));
}

TEST_F(JpegBufferLogNodeTest, DoesNotRecordAcrossQueuedLoggingTransitions) {
  for (bool initially_logging : {false, true}) {
    if (initially_logging) {
      SetEnabled(false);
      SetEnabled(true);  // Second enable, so the next disable stops logging.
    } else {
      SetEnabled(false);
    }
    ASSERT_EQ(control_loop::ShouldLogImages(), initially_logging);
    std::promise<void> started;
    std::promise<void> release;
    auto released = release.get_future();
    pool_.Submit([&]() -> void {
      started.set_value();
      released.wait();
    });
    started.get_future().wait();
    auto done = QueueFrame();
    SetEnabled(!initially_logging);
    release.set_value();
    ASSERT_EQ(done.wait_for(3s), std::future_status::ready);
    EXPECT_FALSE(std::filesystem::exists(FramePath()));
  }
}

TEST_F(JpegBufferLogNodeTest, StandaloneRecordingDoesNotRequireRobotEnable) {
  node_ = std::make_unique<logging::JpegBufferLogNode>(
      "jpeg", folder_.string(), pool_);
  node_->RegisterCallback([this](const control_loop::Context&) -> void {
    completed_.set_value();
  });
  EXPECT_FALSE(control_loop::ShouldLogImages());
  Record(true);
}
}  // namespace
