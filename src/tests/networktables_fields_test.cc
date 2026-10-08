#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <networktables/GenericEntry.h>
#include <unistd.h>

#include "apriltag/tag_detections.h"
#include "camera/cpu_decode_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/timed_node.h"
#include "localization/position.h"
#include "localization/solver_common.h"
#include "logging/wpilog_writer.h"
#include "wpilog_test_utils.h"

namespace {

enum class State : std::uint8_t { tracking = 7 };

struct NativeFields {
  bool ready = true;
  int count = -17;
  float single = 1.25f;
  double real = 2.5;
  std::string text = "tracking";
  std::vector<std::string> strings{"left", "right"};
  std::vector<unsigned> integers{3, 7};
  std::array<float, 2> reals{1.5f, 2.5f};
  std::vector<bool> booleans{true, false};
  frc::Pose2d pose2d{units::meter_t{1}, units::meter_t{2},
                     frc::Rotation2d{units::radian_t{0.5}}};
  frc::Pose3d pose3d{units::meter_t{3}, units::meter_t{4}, units::meter_t{5},
                     frc::Rotation3d{units::radian_t{0.1}, units::radian_t{0.2},
                                     units::radian_t{0.3}}};
  std::vector<frc::Pose2d> poses2d{pose2d, frc::Pose2d{}};
  std::vector<frc::Pose3d> poses3d{pose3d, frc::Pose3d{}};
  std::chrono::milliseconds duration{25};
  State enumeration = State::tracking;
  long double extended = 3.5;
  LOG_FIELDS(NativeFields, ready, count, single, real, text, strings, integers,
             reals, booleans, pose2d, pose3d, poses2d, poses3d, duration,
             enumeration, extended)
};

template <typename T>
struct OptionalField {
  std::optional<T> value;
  LOG_FIELDS(OptionalField, value)
};

struct ExpectedTopic {
  std::string type;
  nt::Value value;
};
using ExpectedTopics = std::unordered_map<std::string, ExpectedTopic>;

template <typename T>
auto StructValue(const T& value) -> nt::Value {
  std::vector<std::uint8_t> bytes(wpi::GetStructSize<T>());
  wpi::PackStruct(bytes, value);
  return nt::Value::MakeRaw(std::move(bytes));
}

template <typename T>
auto StructArrayValue(const std::vector<T>& values) -> nt::Value {
  std::vector<std::uint8_t> bytes(values.size() * wpi::GetStructSize<T>());
  for (std::size_t i = 0; i < values.size(); ++i) {
    wpi::PackStruct(std::span(bytes).subspan(i * wpi::GetStructSize<T>()),
                    values[i]);
  }
  return nt::Value::MakeRaw(std::move(bytes));
}

class NetworkTablesFieldsTest : public ::testing::Test {
 protected:
  // Local retrieval validates registration and publication on the Jetson.
  // It does not establish transmission to a separate device.
  nt::NetworkTableInstance instance_ = nt::NetworkTableInstance::Create();
  std::filesystem::path path_ =
      std::filesystem::temp_directory_path() /
      ("cos-nt-retrieval-" + std::to_string(getpid()) + ".wpilog");
  std::unordered_map<std::string, nt::GenericSubscriber> subscribers_;

  void TearDown() override {
    subscribers_.clear();
    nt::NetworkTableInstance::Destroy(instance_);
    std::filesystem::remove(path_);
  }

  void Subscribe(const ExpectedTopics& expected) {
    // Subscribe to an independent list, rather than discovering the topics
    // created by the writer. A missing publication must fail the test.
    for (const auto& [name, topic] : expected) {
      subscribers_.emplace(
          name, instance_.GetTopic("/COS/" + name).GenericSubscribe(
                    topic.type, {.pollStorage = 10,
                                 .sendAll = true,
                                 .keepDuplicates = true}));
      // This ntcore version queues an unassigned current value when the
      // publisher exists but has not sent a sample yet.
      for (const auto& initial : subscribers_.at(name).ReadQueue()) {
        EXPECT_EQ(initial.type(), NT_UNASSIGNED) << name;
      }
    }
  }

  void CheckTopics(const ExpectedTopics& expected) {
    std::unordered_set<std::string> names;
    for (auto topic : instance_.GetTopics()) {
      if (topic.GetName().starts_with("/.schema/")) continue;
      ASSERT_TRUE(topic.GetName().starts_with("/COS/")) << topic.GetName();
      names.insert(topic.GetName().substr(5));
    }
    std::unordered_set<std::string> expected_names;
    for (const auto& [name, topic] : expected) {
      SCOPED_TRACE(name);
      expected_names.insert(name);
      EXPECT_EQ(instance_.GetTopic("/COS/" + name).GetTypeString(), topic.type);
    }
    EXPECT_EQ(names, expected_names);
  }

  void CheckValues(const ExpectedTopics& expected) {
    for (const auto& [name, topic] : expected) {
      SCOPED_TRACE(name);
      auto& subscriber = subscribers_.at(name);
      const auto latest = subscriber.Get();
      ASSERT_EQ(latest.type(), topic.value.type());
      EXPECT_EQ(latest, topic.value);
      EXPECT_GT(latest.time(), 0);
      const auto queue = subscriber.ReadQueue();
      ASSERT_EQ(queue.size(), 1);
      EXPECT_EQ(queue.front(), topic.value);
      EXPECT_EQ(queue.front().time(), latest.time());
    }
  }

  void CheckNoNewValues() {
    for (auto& [name, subscriber] : subscribers_) {
      EXPECT_TRUE(subscriber.ReadQueue().empty()) << name;
    }
  }
};

TEST_F(NetworkTablesFieldsTest,
       RetrievesEveryNativeFieldWithExpectedTypeAndValue) {
  const NativeFields sample;
  const ExpectedTopics expected{
      {"native/ready",
       {.type = "boolean", .value = nt::Value::MakeBoolean(true)}},
      {"native/count", {.type = "int", .value = nt::Value::MakeInteger(-17)}},
      {"native/single",
       {.type = "float", .value = nt::Value::MakeFloat(1.25f)}},
      {"native/real", {.type = "double", .value = nt::Value::MakeDouble(2.5)}},
      {"native/text",
       {.type = "string", .value = nt::Value::MakeString("tracking")}},
      {"native/strings",
       {.type = "string[]",
        .value = nt::Value::MakeStringArray({"left", "right"})}},
      {"native/integers",
       {.type = "int[]", .value = nt::Value::MakeIntegerArray({3, 7})}},
      {"native/reals",
       {.type = "double[]", .value = nt::Value::MakeDoubleArray({1.5, 2.5})}},
      {"native/booleans",
       {.type = "boolean[]",
        .value = nt::Value::MakeBooleanArray({true, false})}},
      {"native/pose2d",
       {.type = "struct:Pose2d", .value = StructValue(sample.pose2d)}},
      {"native/pose3d",
       {.type = "struct:Pose3d", .value = StructValue(sample.pose3d)}},
      {"native/poses2d",
       {.type = "struct:Pose2d[]", .value = StructArrayValue(sample.poses2d)}},
      {"native/poses3d",
       {.type = "struct:Pose3d[]", .value = StructArrayValue(sample.poses3d)}},
      {"native/duration",
       {.type = "double", .value = nt::Value::MakeDouble(0.025)}},
      {"native/enumeration",
       {.type = "int", .value = nt::Value::MakeInteger(7)}},
      {"native/extended",
       {.type = "double", .value = nt::Value::MakeDouble(3.5)}}};
  Subscribe(expected);
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<NativeFields>("native")};
  logging::WPILogWriter writer(path_.string(), publications, instance_);
  CheckTopics(expected);
  for (int i = 0; i < 2; ++i) {
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, i);
    context.SetMessage(
        "native",
        std::make_unique<control_loop::ValueMessage<NativeFields>>(sample));
    writer.Log(context);
    CheckValues(expected);  // Identical samples must remain retrievable.
  }
  EXPECT_EQ(instance_.GetStructTopic<frc::Pose2d>("/COS/native/pose2d")
                .Subscribe({})
                .Get(),
            sample.pose2d);
  EXPECT_EQ(instance_.GetStructTopic<frc::Pose3d>("/COS/native/pose3d")
                .Subscribe({})
                .Get(),
            sample.pose3d);
  EXPECT_EQ(instance_.GetStructArrayTopic<frc::Pose2d>("/COS/native/poses2d")
                .Subscribe({})
                .Get(),
            sample.poses2d);
  EXPECT_EQ(instance_.GetStructArrayTopic<frc::Pose3d>("/COS/native/poses3d")
                .Subscribe({})
                .Get(),
            sample.poses3d);

  control_loop::ContextInternal empty(std::chrono::steady_clock::now(), nullptr,
                                      std::stop_token{}, 3);
  writer.Log(empty);
  empty.SetMessage("native",
                   std::make_unique<control_loop::ValueMessage<int>>(7));
  EXPECT_THROW(writer.Log(empty), std::runtime_error);
  CheckNoNewValues();
}

TEST_F(NetworkTablesFieldsTest,
       FinalPoseUsesOneCanonicalRepresentationForNTAndWPILog) {
  const frc::Pose3d pose{units::meter_t{1}, units::meter_t{2},
                         units::meter_t{3}, frc::Rotation3d{}};
  const ExpectedTopics expected{
      {"pose_with_variance/pose",
       {.type = "struct:Pose3d", .value = StructValue(pose)}},
      {"pose_with_variance/variance",
       {.type = "double", .value = nt::Value::MakeDouble(0.25)}},
      {"pose_with_variance/timestamp",
       {.type = "double", .value = nt::Value::MakeDouble(12.5)}},
      {"pose_with_variance/tag_ids",
       {.type = "int[]", .value = nt::Value::MakeIntegerArray({7})}},
      {"pose_with_variance/num_tags",
       {.type = "int", .value = nt::Value::MakeInteger(1)}},
      {"pose_with_variance/distances",
       {.type = "double[]", .value = nt::Value::MakeDoubleArray({4})}}};
  // Already-prefixed and absolute channels must not create COS/COS aliases.
  for (const std::string channel :
       {"pose_with_variance", "/pose_with_variance", "COS/pose_with_variance",
        "/COS/pose_with_variance"}) {
    SCOPED_TRACE(channel);
    subscribers_.clear();
    const std::vector publications{
        control_loop::MessageDescriptor::Publication<
            localization::PositionEstimateMessage>(channel)};
    logging::WPILogWriter writer(path_.string(), publications, instance_);
    Subscribe(expected);
    CheckTopics(expected);
    {
      control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                            nullptr, std::stop_token{}, 1);
      auto estimate = std::make_unique<localization::PositionEstimateMessage>();
      estimate->pose = pose;
      estimate->variance = 0.25;
      estimate->timestamp = 12.5;
      estimate->tag_ids = {7};
      estimate->num_tags = 1;
      estimate->distances = {4};
      context.SetMessage(channel, std::move(estimate));
      writer.Log(context);
    }
    CheckValues(expected);
    writer.Flush();
    std::unordered_set<std::string> written;
    wpilog_test::VisitLogValues(path_, [&](const auto& name,
                                           const auto& record) -> void {
      if (name.starts_with("/.schema/")) return;
      ASSERT_TRUE(name.starts_with("/COS/")) << name;
      ASSERT_TRUE(expected.contains(name.substr(5))) << name;
      EXPECT_TRUE(written.insert(name).second) << name;
      const auto value = instance_.GetTopic(name).GenericSubscribe().Get();
      EXPECT_EQ(record.GetTimestamp(), value.time());
      EXPECT_EQ(value.time(), 12'500'000);
      if (name.ends_with("/pose")) {
        EXPECT_EQ(wpi::UnpackStruct<frc::Pose3d>(record.GetRaw()), pose);
      } else if (name.ends_with("/variance") || name.ends_with("/timestamp")) {
        double sample = 0;
        ASSERT_TRUE(record.GetDouble(&sample));
        EXPECT_EQ(sample, value.GetDouble());
      }
    });
    EXPECT_EQ(written.size(), expected.size());
    subscribers_.clear();
  }
}

TEST_F(NetworkTablesFieldsTest, UsesImageTimeAndFallsBackWithoutValidImages) {
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<double>("sample")};
  logging::WPILogWriter writer(path_.string(), publications, instance_);
  auto subscriber = instance_.GetDoubleTopic("/COS/sample").Subscribe(0);
  const std::vector<std::vector<double>> frame_times{
      {0}, {12, 16}, {}, {std::numeric_limits<double>::quiet_NaN(), -1}};
  for (std::size_t i = 0; i < frame_times.size(); ++i) {
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, i);
    for (std::size_t j = 0; j < frame_times[i].size(); ++j) {
      context.SetMessage("jpeg/" + std::to_string(j),
                         std::make_unique<camera::JpegBuffer>(
                             0, frame_times[i][j]));
    }
    context.SetMessage("sample",
                       std::make_unique<control_loop::ValueMessage<double>>(
                           static_cast<double>(i + 1)));
    const auto before = nt::Now();
    writer.Log(context);
    const auto after = nt::Now();
    const auto value = subscriber.GetAtomic();
    EXPECT_EQ(value.value, i + 1);
    if (i == 1) {
      EXPECT_EQ(value.time, 14'000'000);
    } else if (i == 0) {
      EXPECT_EQ(value.time, 1);
    } else {
      EXPECT_GE(value.time, before);
      EXPECT_LE(value.time, after);
    }
  }
}

TEST_F(NetworkTablesFieldsTest,
       RetrievesOptionalPrimitivesAndResetsAbsentValues) {
  const auto check = [&]<typename T>() -> void {
    const std::vector publications{
        control_loop::MessageDescriptor::Publication<OptionalField<T>>(
            "optional")};
    // Each instantiation releases its publisher and subscriber before reusing
    // the same path with a different type.
    subscribers_.clear();
    logging::WPILogWriter writer(path_.string(), publications, instance_);
    for (bool present : {false, true, false}) {
      T populated{};
      if constexpr (std::is_same_v<T, std::string>)
        populated = "present";
      else if constexpr (std::is_same_v<T, frc::Pose3d>) {
        populated = frc::Pose3d{units::meter_t{1}, units::meter_t{2},
                                units::meter_t{3}, frc::Rotation3d{}};
      } else
        populated = static_cast<T>(1);
      const T expected_value = present ? populated : T{};
      ExpectedTopic value;
      if constexpr (std::is_same_v<T, std::string>)
        value = {.type = "string",
                 .value = nt::Value::MakeString(expected_value)};
      else if constexpr (std::is_same_v<T, frc::Pose3d>)
        value = {.type = "struct:Pose3d", .value = StructValue(expected_value)};
      else if constexpr (std::is_same_v<T, bool>)
        value = {.type = "boolean",
                 .value = nt::Value::MakeBoolean(expected_value)};
      else if constexpr (std::is_same_v<T, float>)
        value = {.type = "float",
                 .value = nt::Value::MakeFloat(expected_value)};
      else if constexpr (std::is_floating_point_v<T>)
        value = {.type = "double",
                 .value = nt::Value::MakeDouble(
                     static_cast<double>(expected_value))};
      else
        value = {.type = "int",
                 .value = nt::Value::MakeInteger(
                     static_cast<std::int64_t>(expected_value))};
      const ExpectedTopics expected{
          {"optional/value", value},
          {"optional/value_present",
           {.type = "boolean", .value = nt::Value::MakeBoolean(present)}}};
      if (subscribers_.empty())
        Subscribe(expected);
      CheckTopics(expected);
      control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                            nullptr, std::stop_token{}, 1);
      OptionalField<T> sample;
      if (present)
        sample.value = populated;
      context.SetMessage(
          "optional",
          std::make_unique<control_loop::ValueMessage<OptionalField<T>>>(
              sample));
      writer.Log(context);
      CheckValues(expected);
    }
    subscribers_.clear();
  };
  std::apply(
      [&](const auto&... values) -> void {
        (check.template operator()<std::remove_cvref_t<decltype(values)>>(),
         ...);
      },
      std::tuple<bool, char, signed char, unsigned char, short, unsigned short,
                 int, unsigned int, long, unsigned long, long long,
                 unsigned long long, wchar_t, char8_t, char16_t, char32_t,
                 float, double, long double, State, std::string,
                 frc::Pose3d>{});
}

void AddSolverTopics(ExpectedTopics& expected, const std::string& prefix,
                     const localization::SolverEstimate& sample) {
  expected.emplace(
      prefix + "/tag_ids",
      ExpectedTopic{
          .type = "int[]",
          .value = nt::Value::MakeIntegerArray(std::vector<std::int64_t>(
              sample.tag_ids.begin(), sample.tag_ids.end()))});
  expected.emplace(
      prefix + "/distances",
      ExpectedTopic{.type = "double[]",
                    .value = nt::Value::MakeDoubleArray(sample.distances)});
  expected.emplace(prefix + "/pose",
                   ExpectedTopic{.type = "struct:Pose3d",
                                 .value = StructValue(sample.pose)});
  expected.emplace(
      prefix + "/variance",
      ExpectedTopic{.type = "double",
                    .value = nt::Value::MakeDouble(sample.variance)});
  expected.emplace(
      prefix + "/distance",
      ExpectedTopic{.type = "double",
                    .value = nt::Value::MakeDouble(sample.distance)});
}

TEST_F(NetworkTablesFieldsTest,
       ContextDestructionPublishesEveryProductionField) {
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<camera::JpegBuffer>(
          "camera/jpeg"),
      control_loop::MessageDescriptor::Publication<camera::DecodedImageBuffer>(
          "camera/cpu"),
      control_loop::MessageDescriptor::Publication<camera::DecodedJpegBuffer>(
          "camera/gpu"),
      control_loop::MessageDescriptor::Publication<camera::DecodedJpegFdBuffer>(
          "camera/fd"),
      control_loop::MessageDescriptor::Publication<apriltag::TagDetections>(
          "detections"),
      control_loop::MessageDescriptor::Publication<
          apriltag::TagDetections::tag_detection>("tag"),
      control_loop::MessageDescriptor::Publication<
          localization::SolverEstimate>("solver"),
      control_loop::MessageDescriptor::Publication<
          localization::AmbiguousEstimate>("ambiguous"),
      control_loop::MessageDescriptor::Publication<
          localization::AmbiguousEstimateMessage>("batch"),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>("localization/camera1/pose"),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>("localization/camera2/pose"),
      control_loop::MessageDescriptor::Publication<
          control_loop::LatencyMessage>("decode/camera1/latency")};
  auto writer = std::make_shared<logging::WPILogWriter>(
      path_.string(), publications, instance_);
  const frc::Pose3d left_pose{
      units::meter_t{1}, units::meter_t{2}, units::meter_t{3},
      frc::Rotation3d{units::radian_t{0.1}, units::radian_t{0.2},
                      units::radian_t{0.3}}};
  const frc::Pose3d right_pose{units::meter_t{4}, units::meter_t{5},
                               units::meter_t{6}, frc::Rotation3d{}};
  const localization::SolverEstimate primary{.tag_ids = {1, 2},
                                             .distances = {3.5, 4.5},
                                             .pose = left_pose,
                                             .variance = 0.25,
                                             .distance = 4};
  const localization::SolverEstimate secondary{.tag_ids = {9},
                                               .distances = {7.5},
                                               .pose = right_pose,
                                               .variance = 0.75,
                                               .distance = 8};

  for (bool present : {false, true, false}) {
    ExpectedTopics expected{
        {"camera/jpeg/size",
         {.type = "int", .value = nt::Value::MakeInteger(4)}},
        {"camera/jpeg/timestamp",
         {.type = "double", .value = nt::Value::MakeDouble(12.5)}},
        {"camera/cpu/width",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/cpu/height",
         {.type = "int", .value = nt::Value::MakeInteger(480)}},
        {"camera/cpu/stride",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/gpu/width",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/gpu/height",
         {.type = "int", .value = nt::Value::MakeInteger(480)}},
        {"camera/gpu/stride",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/gpu/output_size",
         {.type = "int", .value = nt::Value::MakeInteger(307200)}},
        {"camera/gpu/output_format",
         {.type = "int", .value = nt::Value::MakeInteger(NVJPEG_OUTPUT_Y)}},
        {"camera/gpu/channel_sizes",
         {.type = "int[]",
          .value = nt::Value::MakeIntegerArray({307200, 0, 0, 0})}},
        {"camera/fd/fd", {.type = "int", .value = nt::Value::MakeInteger(-1)}},
        {"camera/fd/pixel_format",
         {.type = "int", .value = nt::Value::MakeInteger(0x3231564e)}},
        {"camera/fd/width",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/fd/height",
         {.type = "int", .value = nt::Value::MakeInteger(480)}},
        {"camera/fd/stride",
         {.type = "int", .value = nt::Value::MakeInteger(640)}},
        {"camera/fd/output_size",
         {.type = "int", .value = nt::Value::MakeInteger(307200)}},
        {"tag/tag_id", {.type = "int", .value = nt::Value::MakeInteger(9)}},
        {"decode/camera1/latency/latency",
         {.type = "double", .value = nt::Value::MakeDouble(0.012)}}};
    AddSolverTopics(expected, "solver", primary);
    for (const std::string prefix : {"ambiguous", "batch/estimate"}) {
      AddSolverTopics(expected, prefix + "/pos1", primary);
      AddSolverTopics(expected, prefix + "/pos2",
                      present ? secondary : localization::SolverEstimate{});
      expected.emplace(prefix + "/pos2_present",
                       ExpectedTopic{.type = "boolean",
                                     .value = nt::Value::MakeBoolean(present)});
    }
    for (const std::string channel :
         {"localization/camera1/pose", "localization/camera2/pose"}) {
      const bool left = channel == "localization/camera1/pose";
      expected.emplace(
          channel + "/tag_ids",
          ExpectedTopic{.type = "int[]",
                        .value = nt::Value::MakeIntegerArray(
                            left ? std::initializer_list<std::int64_t>{1, 2}
                                 : std::initializer_list<std::int64_t>{9})});
      expected.emplace(
          channel + "/num_tags",
          ExpectedTopic{.type = "int",
                        .value = nt::Value::MakeInteger(left ? 2 : 1)});
      expected.emplace(
          channel + "/distances",
          ExpectedTopic{.type = "double[]",
                        .value = nt::Value::MakeDoubleArray(
                            left ? std::initializer_list<double>{3.5, 4.5}
                                 : std::initializer_list<double>{7.5})});
      expected.emplace(
          channel + "/pose",
          ExpectedTopic{.type = "struct:Pose3d",
                        .value = StructValue(left ? left_pose : right_pose)});
      expected.emplace(
          channel + "/timestamp",
          ExpectedTopic{.type = "double", .value = nt::Value::MakeDouble(12.5)});
      expected.emplace(
          channel + "/variance",
          ExpectedTopic{.type = "double",
                        .value = nt::Value::MakeDouble(left ? 0.25 : 0.75)});
    }
    if (subscribers_.empty())
      Subscribe(expected);
    CheckTopics(expected);
    {
      control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                            nullptr, std::stop_token{}, 1,
                                            writer);
      context.SetMessage("camera/jpeg",
                         std::make_unique<camera::JpegBuffer>(4, 12.5));
      // These frames do not need logging registrations to contribute to
      // capture time. Invalid frame times must not affect the mean.
      context.SetMessage("other/jpeg",
                         std::make_unique<camera::JpegBuffer>(0, 16));
      context.SetMessage("invalid/jpeg",
                         std::make_unique<camera::JpegBuffer>(
                             0, std::numeric_limits<double>::quiet_NaN()));
      auto cpu = std::make_unique<camera::DecodedImageBuffer>();
      cpu->width = 640;
      cpu->height = 480;
      cpu->stride = 640;
      cpu->data = {1, 2, 3};
      context.SetMessage("camera/cpu", std::move(cpu));
      auto gpu = std::make_unique<camera::DecodedJpegBuffer>();
      gpu->width = 640;
      gpu->height = 480;
      gpu->stride = 640;
      gpu->output_size = 307200;
      gpu->channel_sizes[0] = 307200;
      context.SetMessage("camera/gpu", std::move(gpu));
      auto fd = std::make_unique<camera::DecodedJpegFdBuffer>();
      fd->width = 640;
      fd->height = 480;
      fd->stride = 640;
      fd->output_size = 307200;
      fd->pixel_format = 0x3231564e;
      context.SetMessage("camera/fd", std::move(fd));
      const apriltag::TagDetections::tag_detection detection{.tag_id = 9,
                                                             .corners = {}};
      context.SetMessage(
          "tag", std::make_unique<control_loop::ValueMessage<
                     apriltag::TagDetections::tag_detection>>(detection));
      context.SetMessage(
          "detections",
          std::make_unique<apriltag::TagDetections>(std::vector{detection}));
      context.SetMessage(
          "solver",
          std::make_unique<
              control_loop::ValueMessage<localization::SolverEstimate>>(
              primary));
      localization::AmbiguousEstimate ambiguous{.pos1 = primary,
                                                .pos2 = std::nullopt};
      if (present)
        ambiguous.pos2 = secondary;
      context.SetMessage(
          "ambiguous",
          std::make_unique<
              control_loop::ValueMessage<localization::AmbiguousEstimate>>(
              ambiguous));
      context.SetMessage(
          "batch",
          std::make_unique<localization::AmbiguousEstimateMessage>(ambiguous));
      for (const std::string channel :
           {"localization/camera1/pose", "localization/camera2/pose"}) {
        const auto& source =
            channel == "localization/camera1/pose" ? primary : secondary;
        auto pose = std::make_unique<localization::PositionEstimateMessage>();
        pose->tag_ids = source.tag_ids;
        pose->num_tags = static_cast<int>(source.tag_ids.size());
        pose->distances = source.distances;
        pose->pose = source.pose;
        pose->variance = source.variance;
        pose->timestamp = 12.5;
        context.SetMessage(channel, std::move(pose));
      }
      context.SetMessage("decode/camera1/latency",
                         std::make_unique<control_loop::LatencyMessage>(
                             std::chrono::duration<double>(0.012)));
    }  // The destructor is the only call to writer->Log().
    CheckValues(expected);
    for (const auto& [name, topic] : expected) {
      const bool has_own_timestamp = name.starts_with("camera/jpeg/") ||
                                    name.starts_with("localization/");
      EXPECT_EQ(subscribers_.at(name).Get().time(),
                has_own_timestamp ? 12'500'000 : 14'250'000) << name;
    }
  }
  // Image payloads, pointers, detection vectors and corner arrays are omitted.
  EXPECT_FALSE(instance_.GetTopic("/COS/camera/cpu/data").Exists());
  EXPECT_FALSE(instance_.GetTopic("/COS/camera/gpu/destination").Exists());
  EXPECT_FALSE(instance_.GetTopic("/COS/camera/jpeg/ptr").Exists());
  EXPECT_FALSE(instance_.GetTopic("/COS/detections/tag_detections").Exists());
  EXPECT_FALSE(instance_.GetTopic("/COS/tag/corners").Exists());
}

}  // namespace
