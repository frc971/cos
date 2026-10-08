#include "streamer/position_estimate_rio_streamer_node.h"
#include "localization/position.h"

#include "absl/log/log.h"

namespace {
constexpr nt::PubSubOptions publish_options = {.pollStorage = 200,
                                               .sendAll = true,
                                               .keepDuplicates = true};
}

namespace streamer {
PositionEstimateRioStreamerNode::PositionEstimateRioStreamerNode(
    std::string_view input_path, std::string_view networktable_path)
    : input_path_(input_path),
      networktable_path_(networktable_path),
      instance_(nt::NetworkTableInstance::GetDefault()),
      dependencies_(
          {{input_path_, typeid(localization::PositionEstimateMessage)}}),
      publications_() {
  std::shared_ptr<nt::NetworkTable> table =
      instance_.GetTable(networktable_path_);

  nt::DoubleArrayTopic position_estimate_topic =
      table->GetDoubleArrayTopic("PositionEstimate");
  nt::StructTopic<frc::Pose2d> pose2d_topic =
      table->GetStructTopic<frc::Pose2d>("Pose2d");
  nt::StructTopic<frc::Pose3d> pose3d_topic =
      table->GetStructTopic<frc::Pose3d>("Pose3d");

  position_estimate_publisher_ =
      position_estimate_topic.Publish(publish_options);
  pose2d_publisher_ = pose2d_topic.Publish(publish_options);
  pose3d_publisher_ = pose3d_topic.Publish(publish_options);
}

auto PositionEstimateRioStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto PositionEstimateRioStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto PositionEstimateRioStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    if (!context->Exists(input_path_)) {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }
    auto position_estimate_message =
        context->GetMessage<localization::PositionEstimateMessage>(input_path_);
    if (position_estimate_message == nullptr) {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }

    auto pose = position_estimate_message->pose;
    pose3d_publisher_.Set(pose);
    pose2d_publisher_.Set(pose.ToPose2d());

    std::array<double, 5> position_estimate_array{
        pose.X().value(), pose.Y().value(), pose.Rotation().Z().value(),
        position_estimate_message->variance, GetTimestamp(context)};
    position_estimate_publisher_.Set(position_estimate_array);
    instance_.Flush();

    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void PositionEstimateRioStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

void PositionEstimateRioStreamerNode::AddCamera(
    const camera::UVCCameraNode& camera) {
  get_camera_timestamps_.emplace_back(
      [path = camera.GetOutputPath()](
          const control_loop::Context& context) -> std::optional<double> {
        auto jpeg_buffer = context->GetMessage<camera::JpegBuffer>(path);
        if (jpeg_buffer == nullptr) {
          return std::nullopt;
        }
        return jpeg_buffer->timestamp;
      });
}

auto PositionEstimateRioStreamerNode::GetTimestamp(
    const control_loop::Context& context) -> double {
  double total_timestamp = 0;
  int timestamp_num = 0;
  for (const auto& get_camera_timestamp : get_camera_timestamps_) {
    auto maybe_timestamp = get_camera_timestamp(context);
    if (maybe_timestamp.has_value()) {
      total_timestamp += maybe_timestamp.value();
      timestamp_num++;
    }
  }
  if (timestamp_num == 0) {
    LOG(WARNING) << "Failed to get timestamp";
    return 0;
  }
  return total_timestamp / timestamp_num;
}

}  // namespace streamer
