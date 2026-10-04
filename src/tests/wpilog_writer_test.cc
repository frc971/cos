#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <limits>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "control_loop/context.h"
#include "control_loop/control_loop.h"
#include "localization/position.h"
#include "logging/wpilog_writer.h"
#include "wpilog_test_utils.h"

namespace {

auto LogPath() -> std::filesystem::path {
  return std::filesystem::temp_directory_path() /
         ("cos-wpilog-writer-" + std::to_string(getpid()) + ".wpilog");
}

struct Pose2dSample {
  frc::Pose2d pose;
  LOG_FIELDS(Pose2dSample, pose)
};

struct UnregisteredSample {
  int value = 0;
};

struct IntegerSample {
  std::uint64_t value = 0;
  LOG_FIELDS(IntegerSample, value)
};

static_assert(requires { Pose2dSample::RegisterWPILog; });
static_assert(requires { IntegerSample::RegisterWPILog; });

TEST(WPILogWriterTest, WritesRegisteredFieldsAndSkipsMissingMessages) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<double>("temperature"),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>(
          "pose"),
  };

  {
    logging::WPILogWriter writer(path.string(), publications);

    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 1);
    context.SetMessage(
        "temperature",
        std::make_unique<control_loop::ValueMessage<double>>(42.5));
    auto estimate =
        std::make_unique<localization::PositionEstimateMessage>();
    estimate->variance = 0.25;
    estimate->tag_ids = {3, 7};
    estimate->num_tags = 2;
    estimate->distances = {1.5, 2.5};
    context.SetMessage("pose", std::move(estimate));
    writer.Log(context);

    control_loop::ContextInternal empty(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, 2);
    empty.SetMessage("pose", nullptr);
    writer.Log(empty);
    writer.Flush();
  }

  std::unordered_map<std::string, int> value_counts;
  const std::unordered_set<std::string> expected_names = {
      "temperature", "pose/pose", "pose/variance", "pose/tag_ids",
      "pose/num_tags",
      "pose/distances"};
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) -> void {
    if (!expected_names.contains(name)) {
      return;
    }
    ++value_counts[name];
    if (name == "temperature") {
      double value = 0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 42.5);
    } else if (name == "pose/variance") {
      double value = 0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 0.25);
    } else if (name == "pose/num_tags") {
      std::int64_t count = -1;
      ASSERT_TRUE(record.GetInteger(&count));
      EXPECT_EQ(count, 2);
    } else if (name == "pose/tag_ids") {
      std::vector<std::int64_t> ids;
      ASSERT_TRUE(record.GetIntegerArray(&ids));
      EXPECT_EQ(ids, (std::vector<std::int64_t>{3, 7}));
    } else if (name == "pose/distances") {
      std::vector<double> distances;
      ASSERT_TRUE(record.GetDoubleArray(&distances));
      EXPECT_EQ(distances, (std::vector<double>{1.5, 2.5}));
    }
  });
  EXPECT_EQ(value_counts.size(), 6);
  for (const auto& [name, count] : value_counts) {
    EXPECT_EQ(count, 1) << name;
  }
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, FlushesValuesWhileWriterIsActive) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<std::int64_t>("count")};

  const auto has_value = [&](std::int64_t expected) -> bool {
    bool found = false;
    try {
      wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) -> void {
        if (name != "count") return;
        std::int64_t value = 0;
        if (record.GetInteger(&value) && value == expected) found = true;
      });
    } catch (const std::runtime_error&) {
      return false;
    }
    return found;
  };

  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal first(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, 1);
    first.SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<std::int64_t>>(1));
    writer.Log(first);
    writer.Flush();
    EXPECT_TRUE(has_value(1));

    control_loop::ContextInternal second(std::chrono::steady_clock::now(),
                                         nullptr, std::stop_token{}, 2);
    second.SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<std::int64_t>>(2));
    writer.Log(second);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(3);
    while (!has_value(2) && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_TRUE(has_value(2));
  }
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, ControlLoopUsesExistingWriterAndRetainedContexts) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<int>("count")};
  auto writer =
      std::make_shared<logging::WPILogWriter>(path.string(), publications);
  {
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                         nullptr, std::stop_token{}, 1);
    context.SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<int>>(1));
    writer->Log(context);
  }

  std::promise<control_loop::Context> retained_context;
  auto future = retained_context.get_future();
  control_loop::ControlLoop loop(std::chrono::milliseconds(1));
  loop.EnableWPILog(writer);
  loop.RegisterCallback([&](const control_loop::Context& context) -> void {
    context->SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<int>>(2));
    retained_context.set_value(context);
  });
  loop.Start();
  const auto status = future.wait_for(std::chrono::seconds(5));
  loop.Stop();
  ASSERT_EQ(status, std::future_status::ready);

  // A context held past Stop() keeps the supplied writer alive.
  const std::weak_ptr<logging::WPILogWriter> weak_writer = writer;
  writer.reset();
  EXPECT_FALSE(weak_writer.expired());
  auto context = future.get();
  context.reset();
  EXPECT_TRUE(weak_writer.expired());

  std::vector<std::int64_t> values;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) -> void {
    if (name != "count") return;
    std::int64_t value = 0;
    ASSERT_TRUE(record.GetInteger(&value));
    values.push_back(value);
  });
  EXPECT_EQ(values, (std::vector<std::int64_t>{1, 2}));
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, RequiresRegistrationForClassPublications) {
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<UnregisteredSample>(
          "unregistered"),
  };
  EXPECT_THROW(logging::WPILogWriter(LogPath().string(), publications),
               std::invalid_argument);
  std::filesystem::remove(LogPath());
}

TEST(WPILogWriterTest, RejectsDuplicateChannelsAndPaths) {
  const auto path = LogPath();
  const std::vector duplicate_channels{
      control_loop::MessageDescriptor::Publication<int>("sample"),
      control_loop::MessageDescriptor::Publication<double>("sample")};
  EXPECT_THROW(logging::WPILogWriter(path.string(), duplicate_channels),
               std::invalid_argument);

  const std::vector duplicate_paths{
      control_loop::MessageDescriptor::Publication<int>("sample/value"),
      control_loop::MessageDescriptor::Publication<IntegerSample>("sample")};
  EXPECT_THROW(logging::WPILogWriter(path.string(), duplicate_paths),
               std::invalid_argument);
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, PrimitiveAndAnnotatedIntegersShareCheckedConversion) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<std::uint64_t>("primitive"),
      control_loop::MessageDescriptor::Publication<IntegerSample>("annotated")};
  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal valid(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, 1);
    const auto max = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max());
    valid.SetMessage(
        "primitive",
        std::make_unique<control_loop::ValueMessage<std::uint64_t>>(max));
    valid.SetMessage(
        "annotated",
        std::make_unique<control_loop::ValueMessage<IntegerSample>>(
            IntegerSample{max}));
    EXPECT_NO_THROW(writer.Log(valid));

    control_loop::ContextInternal overflow(std::chrono::steady_clock::now(),
                                           nullptr, std::stop_token{}, 2);
    overflow.SetMessage(
        "primitive",
        std::make_unique<control_loop::ValueMessage<std::uint64_t>>(max + 1));
    EXPECT_THROW(writer.Log(overflow), std::out_of_range);
    overflow.SetMessage("primitive", nullptr);
    overflow.SetMessage(
        "annotated",
        std::make_unique<control_loop::ValueMessage<IntegerSample>>(
            IntegerSample{max + 1}));
    EXPECT_THROW(writer.Log(overflow), std::out_of_range);
  }

  std::unordered_map<std::string, int> counts;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) -> void {
    if (name != "primitive" && name != "annotated/value") return;
    std::int64_t value = 0;
    ASSERT_TRUE(record.GetInteger(&value));
    EXPECT_EQ(value, std::numeric_limits<std::int64_t>::max());
    ++counts[name];
  });
  EXPECT_EQ(counts["primitive"], 1);
  EXPECT_EQ(counts["annotated/value"], 1);
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, WritesBuiltInTypesAndPose2d) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<bool>("ready"),
      control_loop::MessageDescriptor::Publication<std::int64_t>("count"),
      control_loop::MessageDescriptor::Publication<std::string>("state"),
      control_loop::MessageDescriptor::Publication<Pose2dSample>("location"),
  };
  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 3);
    context.SetMessage(
        "ready", std::make_unique<control_loop::ValueMessage<bool>>(true));
    context.SetMessage(
        "count",
        std::make_unique<control_loop::ValueMessage<std::int64_t>>(17));
    context.SetMessage("state",
                       std::make_unique<control_loop::ValueMessage<std::string>>(
                           "tracking"));
    context.SetMessage(
        "location",
        std::make_unique<control_loop::ValueMessage<Pose2dSample>>(
            Pose2dSample{}));
    writer.Log(context);
  }

  std::unordered_set<std::string> seen;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) -> void {
      if (name == "ready") {
        bool value = false;
        ASSERT_TRUE(record.GetBoolean(&value));
        EXPECT_TRUE(value);
      } else if (name == "count") {
        std::int64_t value = 0;
        ASSERT_TRUE(record.GetInteger(&value));
        EXPECT_EQ(value, 17);
      } else if (name == "state") {
        std::string_view value;
        ASSERT_TRUE(record.GetString(&value));
        EXPECT_EQ(value, "tracking");
      } else if (name == "location/pose") {
        EXPECT_FALSE(record.GetRaw().empty());
      } else {
        return;
      }
      EXPECT_TRUE(seen.insert(name).second);
  });
  EXPECT_EQ(seen.size(), 4);
  std::filesystem::remove(path);
}

}  // namespace
