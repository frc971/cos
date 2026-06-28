#include "localization/networktable_sender.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include "absl/log/log.h"

namespace localization {
namespace {

constexpr int kmax_tags = 50;

auto TimestampFromMetadata(const control_loops::MetaDataList& metadata)
    -> int64_t {
  double timestamp = 0.0;
  for (const control_loops::MetaData& entry : metadata) {
    timestamp = std::max(timestamp, entry.timestamp);
  }
  return static_cast<int64_t>(std::llround(timestamp * 1'000'000.0));
}

}  // namespace

NetworkTableSender::NetworkTableSender(const std::string& camera_name,
                                       bool verbose)
    : instance_(wpi::nt::NetworkTableInstance::GetDefault()),
      table_(instance_.GetTable("Orin/PoseEstimate/" + camera_name)),
      verbose_(verbose) {
  pose_publisher_ = table_->GetStructTopic<wpi::math::Pose2d>("Pose").Publish();
  pose3d_publisher_ =
      table_->GetStructTopic<wpi::math::Pose3d>("Pose3d").Publish();
  tag_estimation_publisher_ =
      table_->GetDoubleArrayTopic("TagEstimation").Publish();
  tag_ids_publisher_ = table_->GetBooleanArrayTopic("TagId").Publish();
  rejected_tag_ids_publisher_ =
      table_->GetBooleanArrayTopic("RejectedTagId").Publish();
  num_tags_publisher_ = table_->GetIntegerTopic("NumTags").Publish();
  variance_publisher_ = table_->GetDoubleTopic("Variance").Publish();
  avg_tag_dist_publisher_ = table_->GetDoubleTopic("AvgTagDist").Publish();
  loss_publisher_ = table_->GetDoubleTopic("Loss").Publish();
}

void NetworkTableSender::Send(const position_estimate_t& estimate,
                              control_loops::MetaDataList metadata,
                              std::shared_ptr<control_loops::Context> /*ctx*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int64_t timestamp = TimestampFromMetadata(metadata);

  const auto pose2d = estimate.pose.ToPose2d();
  pose_publisher_.Set(pose2d, timestamp);
  pose3d_publisher_.Set(estimate.pose, timestamp);

  std::array<double, 7> tag_estimation{
      estimate.pose.X().value(),
      estimate.pose.Y().value(),
      estimate.pose.Rotation().Z().value(),
      estimate.variance,
      static_cast<double>(estimate.num_tags),
      estimate.avg_tag_dist,
      estimate.loss,
  };
  tag_estimation_publisher_.Set(tag_estimation, timestamp);

  std::array<int, kmax_tags> tags{};
  for (const int tag_id : estimate.tag_ids) {
    if (tag_id >= 0 && tag_id < kmax_tags) {
      tags[tag_id] = true;
    }
  }

  std::array<int, kmax_tags> rejected_tags{};
  for (const int tag_id : estimate.rejected_tag_ids) {
    if (tag_id >= 0 && tag_id < kmax_tags) {
      rejected_tags[tag_id] = true;
    }
  }

  tag_ids_publisher_.Set(tags, timestamp);
  rejected_tag_ids_publisher_.Set(rejected_tags, timestamp);
  num_tags_publisher_.Set(estimate.num_tags, timestamp);
  variance_publisher_.Set(estimate.variance, timestamp);
  avg_tag_dist_publisher_.Set(estimate.avg_tag_dist, timestamp);
  loss_publisher_.Set(estimate.loss, timestamp);

  if (verbose_) {
    LOG(INFO) << estimate;
  }
}

auto NetworkTableSender::MakeDoubleChannel(const std::string& subkey)
    -> std::function<void(double)> {
  std::lock_guard<std::mutex> lock(mutex_);
  auto publisher = std::make_shared<wpi::nt::DoublePublisher>(
      table_->GetDoubleTopic(subkey).Publish());
  double_publishers_.push_back(publisher);
  return [publisher](double value) {
    publisher->Set(value);
  };
}

auto NetworkTableSender::MakeBoolChannel(const std::string& subkey)
    -> std::function<void(bool)> {
  std::lock_guard<std::mutex> lock(mutex_);
  auto publisher = std::make_shared<wpi::nt::BooleanPublisher>(
      table_->GetBooleanTopic(subkey).Publish());
  bool_publishers_.push_back(publisher);
  return [publisher](bool value) {
    publisher->Set(value);
  };
}

auto NetworkTableSender::MakeStringChannel(const std::string& subkey)
    -> std::function<void(std::string)> {
  std::lock_guard<std::mutex> lock(mutex_);
  auto publisher = std::make_shared<wpi::nt::StringPublisher>(
      table_->GetStringTopic(subkey).Publish());
  string_publishers_.push_back(publisher);
  return [publisher](std::string value) {
    publisher->Set(std::move(value));
  };
}

}  // namespace localization
