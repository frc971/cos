#include "streamer/densest_cluster_rio_streamer_node.h"

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "localization/position.h"
#include "streamer/cluster_geometry.h"
#include "streamer/cluster_rio_streamer_node.h"

namespace streamer {

DensestClusterRioStreamerNode::DensestClusterRioStreamerNode(
    std::string_view input_path, std::string_view pose_path,
    std::string_view networktable_path, nt::NetworkTableInstance instance)
    : input_path_(input_path),
      pose_path_(pose_path),
      output_path_(networktable_path),
      dependencies_(
          {{input_path_,
            typeid(control_loop::ValueMessage<gamepiece::hsv_clusters_t>)},
           {pose_path_, typeid(localization::PositionEstimateMessage)}}),
      publications_(
          {control_loop::MessageDescriptor::Publication<std::vector<double>>(
              output_path_)}) {
  if (input_path_.empty() || pose_path_.empty() || output_path_.empty() ||
      input_path_ == pose_path_ || input_path_ == output_path_ ||
      pose_path_ == output_path_) {
    throw std::invalid_argument("Cluster sender requires three distinct paths");
  }
  publisher_ =
      instance.GetDoubleArrayTopic(output_path_)
          .Publish({.periodic = ClusterRioStreamerNode::kTransmitPeriodSeconds,
                    .sendAll = true,
                    .keepDuplicates = true});
}

auto DensestClusterRioStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    {
      std::scoped_lock lock(mutex_);
      if (!context->Exists(input_path_) || !context->Exists(pose_path_) ||
          context->Exists(output_path_))
        return;
      const auto* clusters = context->GetMessage<
          control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(input_path_);
      const auto* estimate =
          context->GetMessage<localization::PositionEstimateMessage>(
              pose_path_);
      if (clusters == nullptr) {
        context->SetMessage(output_path_, nullptr);
      } else {
        std::optional<gamepiece::cluster_nt_t> best;
        if (estimate != nullptr) {
          const auto pose = estimate->pose.ToPose2d();
          for (const auto& cluster : clusters->value) {
            auto candidate = detail::FieldCluster(cluster, pose);
            if (candidate && (!best || candidate->predicted_density >
                                           best->predicted_density)) {
              best = candidate;
            }
          }
        }
        std::vector<double> values;
        if (best) {
          const auto& c = *best;
          const double yaw = c.field_relative_semimajor_axis_yaw;
          const double major_x = c.semimajor_length * std::cos(yaw);
          const double major_y = c.semimajor_length * std::sin(yaw);
          const double minor_x = -c.semiminor_length * std::sin(yaw);
          const double minor_y = c.semiminor_length * std::cos(yaw);
          values = {c.center_x,           c.center_y,
                    c.center_x + major_x, c.center_y + major_y,
                    c.center_x - major_x, c.center_y - major_y,
                    c.center_x + minor_x, c.center_y + minor_y,
                    c.center_x - minor_x, c.center_y - minor_y};
        }
        publisher_.Set(values);
        context->SetMessage(
            output_path_,
            std::make_shared<control_loop::ValueMessage<std::vector<double>>>(
                std::move(values)));
      }
    }
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

void DensestClusterRioStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto DensestClusterRioStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto DensestClusterRioStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace streamer
