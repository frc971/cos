#include "localization/unambiguous_solver_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <frc/geometry/Quaternion.h>
#include <frc/geometry/Rotation3d.h>

#include "absl/log/log.h"
#include "camera/jpeg_buffer.h"
#include "control_loop/control_loop.h"

namespace {

const std::unordered_set<std::type_index> dependency_messages = {
    typeid(localization::AmbiguousEstimateMessage)};

auto FilterOffFieldCandidates(localization::ambiguous_estimate_t* estimate)
    -> bool {
  const bool first_off_field = localization::PoseOffField(estimate->pos1.pose);
  if (!estimate->pos2.has_value()) {
    if (first_off_field) {
      LOG(WARNING) << "Rejecting physically impossible pose: "
                   << estimate->pos1;
    }
    return !first_off_field;
  }

  const bool second_off_field =
      localization::PoseOffField(estimate->pos2->pose);
  if (first_off_field && second_off_field) {
    LOG(WARNING) << "Rejecting two physically impossible pose candidates: "
                 << estimate->pos1 << "; " << *estimate->pos2;
    return false;
  }
  if (first_off_field) {
    VLOG(1) << "Rejecting physically impossible pose candidate: "
            << estimate->pos1;
    estimate->pos1 = std::move(*estimate->pos2);
    estimate->pos2.reset();
  } else if (second_off_field) {
    VLOG(1) << "Rejecting physically impossible pose candidate: "
            << *estimate->pos2;
    estimate->pos2.reset();
  }
  return true;
}
}  // namespace

namespace localization {
using std::weak_ptr;

UnambiguousSolverNode::UnambiguousSolverNode(std::string_view output_channel,
                                             frc::AprilTagFieldLayout layout)
    : output_channel_(output_channel),
      layout_(std::move(layout)),
      publications_({control_loop::MessageDescriptor::Publication<
          PositionEstimateMessage>(output_channel_)}) {}

void UnambiguousSolverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

void UnambiguousSolverNode::AddCameraTimestamp(std::string_view jpeg_channel) {
  camera_timestamp_channels_.emplace_back(jpeg_channel);
}

void UnambiguousSolverNode::AddCamera(std::string_view input_channel,
                                      const camera::Intrinsics& intrinsics,
                                      const camera::Extrinsics& extrinsics,
                                      control_loop::ControlLoop& control_loop) {
  std::string multitag_output_channel =
      std::string(input_channel) + ":multitag_solver";
  auto multitag_solver = std::make_shared<MultiTagSolverNode>(
      input_channel, multitag_output_channel, intrinsics, extrinsics, layout_);
  multitag_solver->SetRejectFarTags(reject_far_tags_);
  multitag_solvers_.push_back(multitag_solver);
  multi_tag_solver_output_channels_.push_back(multitag_output_channel);
  control_loop.RegisterNode(multitag_solver);
  dependencies_.emplace_back(multitag_output_channel, dependency_messages);
}

auto UnambiguousSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    std::scoped_lock lock(solve_mutex_);
    std::vector<ambiguous_estimate_t*> estimates;
    for (const auto& multi_tag_solver_output_channel :
         multi_tag_solver_output_channels_) {
      if (!context->Exists(multi_tag_solver_output_channel)) {
        return;
      }
      auto ambiguous_estimate = context->GetMessage<AmbiguousEstimateMessage>(
          multi_tag_solver_output_channel);
      if (ambiguous_estimate == nullptr) {
        continue;
      }
      estimates.push_back(&ambiguous_estimate->estimate);
    }
    auto result = Solve(estimates, reject_far_tags_);
    if (result.has_value()) {
      // All camera frames are already in the context when their solvers finish.
      double total_timestamp = 0;
      std::size_t count = 0;
      for (const auto& channel : camera_timestamp_channels_) {
        const auto* frame = context->GetMessage<camera::JpegBuffer>(channel);
        if (frame != nullptr && std::isfinite(frame->timestamp)) {
          total_timestamp += frame->timestamp;
          ++count;
        }
      }
      if (count != 0) result->timestamp = total_timestamp / count;
      context->SetMessage(
          output_channel_,
          std::make_unique<PositionEstimateMessage>(result.value()));
    } else {
      context->SetMessage(output_channel_, nullptr);
    }

    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto UnambiguousSolverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto UnambiguousSolverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void UnambiguousSolverNode::SetRejectFarTags(bool reject_far_tags) {
  reject_far_tags_ = reject_far_tags;
  for (const auto& multitag_solver : multitag_solvers_) {
    multitag_solver->SetRejectFarTags(reject_far_tags);
  }
}

auto UnambiguousSolverNode::Cost(const frc::Pose3d& a, const frc::Pose3d& b)
    -> double {
  const double translation = a.Translation().Distance(b.Translation()).value();
  const frc::Rotation3d delta = a.Rotation().RelativeTo(b.Rotation());
  constexpr double kRotationWeight = 0.1;
  return translation + kRotationWeight * delta.Angle().value();
}

auto UnambiguousSolverNode::ComputeCost(
    const std::vector<solver_estimate_t>& poses) -> double {
  double cost = 0.0;
  for (size_t i = 0; i < poses.size(); ++i) {
    for (size_t j = i + 1; j < poses.size(); ++j) {
      cost += Cost(poses[i].pose, poses[j].pose);
    }
    if (prev_pose_estimate_.has_value()) {
      cost += Cost(poses[i].pose, prev_pose_estimate_->pose);
    }
  }
  return cost;
}

auto UnambiguousSolverNode::WeightedAveragePose(
    const std::vector<solver_estimate_t>& solutions) -> frc::Pose3d {
  if (solutions.empty()) {
    return frc::Pose3d{};
  }
  if (solutions.size() == 1) {
    return solutions.front().pose;
  }

  double total_weight = 0.0;
  for (const auto& estimate : solutions) {
    total_weight += 1.0 / estimate.variance;
  }

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;

  for (const auto& estimate : solutions) {
    const double weight = (1.0 / estimate.variance) / total_weight;
    x += weight * estimate.pose.X().value();
    y += weight * estimate.pose.Y().value();
    z += weight * estimate.pose.Z().value();

    auto quaternion = estimate.pose.Rotation().GetQuaternion();
    const double dot = qw * quaternion.W() + qx * quaternion.X() +
                       qy * quaternion.Y() + qz * quaternion.Z();
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    qw += weight * sign * quaternion.W();
    qx += weight * sign * quaternion.X();
    qy += weight * sign * quaternion.Y();
    qz += weight * sign * quaternion.Z();
  }

  const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  qw /= norm;
  qx /= norm;
  qy /= norm;
  qz /= norm;

  return frc::Pose3d{units::meter_t{x}, units::meter_t{y}, units::meter_t{z},
                     frc::Rotation3d{frc::Quaternion{qw, qx, qy, qz}}};
}

auto UnambiguousSolverNode::SearchSolutions(
    const std::vector<ambiguous_estimate_t*>& all_pose_estimates, size_t index,
    std::vector<solver_estimate_t>& current_solution,
    std::vector<solver_estimate_t>& best_solution, double& best_cost)
    -> double {
  if (index == all_pose_estimates.size()) {
    const double cost = ComputeCost(current_solution);
    if (cost < best_cost) {
      best_cost = cost;
      best_solution = current_solution;
    }
    return best_cost;
  }

  const ambiguous_estimate_t* maybe_ambiguous = all_pose_estimates[index];
  current_solution.push_back(maybe_ambiguous->pos1);
  SearchSolutions(all_pose_estimates, index + 1, current_solution,
                  best_solution, best_cost);
  current_solution.pop_back();

  if (maybe_ambiguous->pos2.has_value()) {
    current_solution.push_back(*maybe_ambiguous->pos2);
    SearchSolutions(all_pose_estimates, index + 1, current_solution,
                    best_solution, best_cost);
    current_solution.pop_back();
  }
  return best_cost;
}

auto UnambiguousSolverNode::GetAmbiguousEstimates(
    const std::vector<std::vector<tag_detection_t>>& detection_batches,
    bool reject_far_tags) -> std::vector<ambiguous_estimate_t> {
  std::vector<ambiguous_estimate_t> estimates;
  const size_t num_cameras =
      std::min(multitag_solvers_.size(), detection_batches.size());
  for (size_t i = 0; i < num_cameras; ++i) {
    if (detection_batches[i].empty()) {
      continue;
    }
    auto estimate = multitag_solvers_[i]->AmbiguousSolve(detection_batches[i],
                                                         reject_far_tags);
    if (!estimate.has_value()) {
      continue;
    }

    if (reject_far_tags && !FilterOffFieldCandidates(&*estimate)) {
      continue;
    }

    estimates.push_back(std::move(*estimate));
  }
  return estimates;
}

auto UnambiguousSolverNode::Solve(
    const std::vector<ambiguous_estimate_t*>& estimates, bool reject_far_tags)
    -> std::optional<position_estimate_t> {
  std::vector<ambiguous_estimate_t> filtered_estimates;
  filtered_estimates.reserve(estimates.size());
  for (const ambiguous_estimate_t* estimate : estimates) {
    if (estimate == nullptr) {
      continue;
    }
    filtered_estimates.push_back(*estimate);
    if (reject_far_tags &&
        !FilterOffFieldCandidates(&filtered_estimates.back())) {
      filtered_estimates.pop_back();
    }
  }

  std::vector<ambiguous_estimate_t*> filtered_estimate_ptrs;
  filtered_estimate_ptrs.reserve(filtered_estimates.size());
  for (ambiguous_estimate_t& estimate : filtered_estimates) {
    filtered_estimate_ptrs.push_back(&estimate);
  }

  std::vector<solver_estimate_t> best_solution;
  std::vector<solver_estimate_t> current_solution;
  double best_cost = std::numeric_limits<double>::infinity();
  SearchSolutions(filtered_estimate_ptrs, 0, current_solution, best_solution,
                  best_cost);

  if (best_solution.empty()) {
    return std::nullopt;
  }
  std::vector<int> tag_ids;
  std::vector<double> distances;
  for (const solver_estimate_t& estimate : best_solution) {
    tag_ids.insert(tag_ids.end(), estimate.tag_ids.begin(),
                   estimate.tag_ids.end());
    distances.insert(distances.end(), estimate.distances.begin(),
                     estimate.distances.end());
  }
  position_estimate_t estimate;
  estimate.num_tags = tag_ids.size();
  estimate.tag_ids = std::move(tag_ids);
  estimate.pose = WeightedAveragePose(best_solution);
  estimate.distances = std::move(distances);
  if (reject_far_tags && PoseOffField(estimate.pose)) {
    LOG(WARNING) << "Rejecting physically impossible combined pose: "
                 << estimate;
    return std::nullopt;
  }
  prev_pose_estimate_.emplace(estimate);
  return estimate;
}

}  // namespace localization
