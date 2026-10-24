#include "streamer/cluster_rio_streamer_node.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/position.h"
#include "streamer/cluster_geometry.h"

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
          for (const auto& cluster : clusters->value) {
            if (auto value = detail::FieldCluster(cluster, pose)) {
              values.push_back(*value);
            }
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
