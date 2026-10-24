#include "streamer/cluster_rio_streamer_node.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/position.h"

namespace streamer {

ClusterRioStreamerNode::ClusterRioStreamerNode(
    std::string_view input_path, std::string_view pose_path,
    std::string_view networktable_path, nt::NetworkTableInstance instance)
    : input_path_(input_path),
      pose_path_(pose_path),
      output_path_(networktable_path),
      dependencies_(
          {{input_path_,
            typeid(control_loop::ValueMessage<gamepiece::hsv_clusters_t>)},
           {pose_path_, typeid(localization::PositionEstimateMessage)}}),
      publications_({control_loop::MessageDescriptor::Publication<
                     std::vector<gamepiece::cluster_nt_t>>(output_path_)}) {
  if (input_path_.empty() || pose_path_.empty() || output_path_.empty() ||
      input_path_ == pose_path_ || input_path_ == output_path_ ||
      pose_path_ == output_path_) {
    throw std::invalid_argument("Cluster sender requires three distinct paths");
  }
  publisher_ = instance.GetStructArrayTopic<gamepiece::cluster_nt_t>(output_path_)
                   .Publish({.periodic = kTransmitPeriodSeconds,
                             .sendAll = true,
                             .keepDuplicates = true});
}

auto ClusterRioStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    {
      std::scoped_lock lock(mutex_);
      if (!context->Exists(input_path_) || !context->Exists(pose_path_) ||
          context->Exists(output_path_)) {
        return;
      }
      const auto* clusters = context->GetMessage<
          control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(input_path_);
      const auto* estimate =
          context->GetMessage<localization::PositionEstimateMessage>(pose_path_);
      if (clusters == nullptr) {
        // No new camera frame. Do not manufacture an empty observation.
        context->SetMessage(output_path_, nullptr);
      } else {
        std::vector<gamepiece::cluster_nt_t> values;
        if (estimate != nullptr) {
          const auto pose = estimate->pose.ToPose2d();
          const double heading = pose.Rotation().Radians().value();
          for (const auto& cluster : clusters->value) {
            if (!cluster.robot_relative_translation) continue;
            const auto center = cluster.robot_relative_translation->RotateBy(
                                    pose.Rotation()) + pose.Translation();
            const double xx = cluster.ground_covar(0, 0);
            const double yy = cluster.ground_covar(1, 1);
            const double xy = 0.5 * (cluster.ground_covar(0, 1) +
                                     cluster.ground_covar(1, 0));
            const double radius = std::hypot(xx - yy, 2 * xy);
            const double major_variance = 0.5 * (xx + yy + radius);
            const double minor_variance = 0.5 * (xx + yy - radius);
            if (!std::isfinite(center.X().value()) ||
                !std::isfinite(center.Y().value()) || !std::isfinite(heading) ||
                !std::isfinite(major_variance) ||
                !std::isfinite(minor_variance) ||
                minor_variance < -1e-6 * std::max(1.0, major_variance) ||
                !std::isfinite(cluster.density) || cluster.density < 0) {
              continue;
            }
            double yaw = std::fmod(0.5 * std::atan2(2 * xy, xx - yy) + heading,
                                   std::numbers::pi);
            if (yaw < 0) yaw += std::numbers::pi;
            values.push_back({
                .center_x = center.X().value(),
                .center_y = center.Y().value(),
                .field_relative_semimajor_axis_yaw = yaw,
                .semimajor_length =
                    2 * std::sqrt(std::max(0.0, major_variance)),
                .semiminor_length =
                    2 * std::sqrt(std::max(0.0, minor_variance)),
                .predicted_density = cluster.density});
          }
        }
        // A real empty frame or unavailable pose clears stale field targets.
        publisher_.Set(values);
        context->SetMessage(
            output_path_, std::make_shared<control_loop::ValueMessage<
                              std::vector<gamepiece::cluster_nt_t>>>(
                              std::move(values)));
      }
    }
    for (const auto& callback : callbacks_) callback(context);
  };
}

void ClusterRioStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto ClusterRioStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto ClusterRioStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace streamer
