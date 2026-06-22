#include "localization/unambiguous_solver_node.h"
#include <cmath>
#include <limits>
#include <unordered_set>
#include <wpi/math/geometry/Quaternion.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>
#include "utils/transform.h"

namespace localization {

UnambiguousSolverNode::UnambiguousSolverNode(
    const std::vector<camera::camera_constant_t>& camera_constants,
    const wpi::apriltag::AprilTagFieldLayout& layout) {
  multitag_solvers_.reserve(camera_constants.size());
  for (const camera::camera_constant_t& cc : camera_constants) {
    camera_names_.push_back(cc.name);
    multitag_solvers_.emplace_back(cc, layout);
  }
}

void UnambiguousSolverNode::RegisterCallback(
    const std::function<void(std::optional<position_estimate_t>)>& callback) {
  callbacks_.push_back(callback);
}

auto UnambiguousSolverNode::Cost(const wpi::math::Pose3d& a,
                                 const wpi::math::Pose3d& b) -> double {
  double translation = a.Translation().Distance(b.Translation()).value();
  wpi::math::Rotation3d delta = a.Rotation().RelativeTo(b.Rotation());
  double rotation = delta.Angle().value();
  constexpr double krotation_weight = 0.1;
  return translation + krotation_weight * rotation;
}

auto UnambiguousSolverNode::ComputeCost(
    const std::vector<position_estimate_t>& poses) -> double {
  double cost = 0.0;
  for (size_t i = 0; i < poses.size(); i++) {
    if (poses[i].invalid) {
      return 1000;
    }
    for (size_t j = i + 1; j < poses.size(); j++) {
      cost += Cost(poses[i].pose, poses[j].pose);
    }
    if (prev_pose_estimate_.has_value()) {
      cost += Cost(poses[i].pose, prev_pose_estimate_.value().pose);
    }
  }
  return cost;
}

auto UnambiguousSolverNode::WeightedAveragePose(
    const std::vector<position_estimate_t>& solutions) -> wpi::math::Pose3d {
  if (solutions.empty()) {
    return wpi::math::Pose3d{};
  }
  if (solutions.size() == 1) {
    return solutions[0].pose;
  }

  double total_weight = 0.0;
  for (const auto& est : solutions) {
    total_weight += 1.0 / est.variance;
  }

  double x = 0, y = 0, z = 0;
  double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;

  for (const auto& est : solutions) {
    double w = (1.0 / est.variance) / total_weight;
    x += w * est.pose.X().value();
    y += w * est.pose.Y().value();
    z += w * est.pose.Z().value();

    auto q = est.pose.Rotation().GetQuaternion();
    if (qw * q.W() + qx * q.X() + qy * q.Y() + qz * q.Z() < 0.0) {
      qw += w * (-q.W());
      qx += w * (-q.X());
      qy += w * (-q.Y());
      qz += w * (-q.Z());
    } else {
      qw += w * q.W();
      qx += w * q.X();
      qy += w * q.Y();
      qz += w * q.Z();
    }
  }

  double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  qw /= norm;
  qx /= norm;
  qy /= norm;
  qz /= norm;

  return wpi::math::Pose3d{
      wpi::units::meter_t{x}, wpi::units::meter_t{y}, wpi::units::meter_t{z},
      wpi::math::Rotation3d{wpi::math::Quaternion{qw, qx, qy, qz}}};
}

auto UnambiguousSolverNode::SearchSolutions(
    const std::vector<ambiguous_estimate_t>& all_pose_estimates, size_t index,
    std::vector<position_estimate_t>& current_solution,
    std::vector<position_estimate_t>& best_solution,
    double& best_cost) -> double {
  if (index == all_pose_estimates.size()) {
    double cost = ComputeCost(current_solution);
    if (cost < best_cost) {
      best_cost = cost;
      best_solution = current_solution;
    }
    return best_cost;
  }

  const ambiguous_estimate_t& maybe_ambiguous = all_pose_estimates[index];
  current_solution.push_back(maybe_ambiguous.pos1);
  SearchSolutions(all_pose_estimates, index + 1, current_solution,
                  best_solution, best_cost);
  current_solution.pop_back();

  if (maybe_ambiguous.pos2.has_value()) {
    current_solution.push_back(maybe_ambiguous.pos2.value());
    SearchSolutions(all_pose_estimates, index + 1, current_solution,
                    best_solution, best_cost);
    current_solution.pop_back();
  }
  return best_cost;
}

auto UnambiguousSolverNode::GetAmbiguousEstimates(
    const std::vector<std::vector<apriltag::tag_detection_t>>&
        detection_batches,
    bool reject_far_tags) -> std::vector<ambiguous_estimate_t> {
  double latest_timestamp = -1;
  for (const auto& detections : detection_batches) {
    if (!detections.empty() && detections[0].timestamp > latest_timestamp) {
      latest_timestamp = detections[0].timestamp;
    }
  }

  std::vector<ambiguous_estimate_t> estimates;
  const size_t num_cameras =
      std::min(multitag_solvers_.size(), detection_batches.size());
  for (size_t i = 0; i < num_cameras; i++) {
    const std::vector<apriltag::tag_detection_t>& detections =
        detection_batches[i];
    if (detections.empty()) {
      continue;
    }
    if (latest_timestamp - detections[0].timestamp >=
        kacceptable_frame_recency) {
      continue;
    }

    std::optional<ambiguous_estimate_t> est =
        multitag_solvers_[i].AmbiguousSolveWithoutNotify(detections,
                                                         reject_far_tags);
    if (!est.has_value()) {
      continue;
    }

    bool first_off_field = utils::PoseOffField(est->pos1.pose);
    if (est->pos2.has_value()) {
      bool second_off_field = utils::PoseOffField(est->pos2.value().pose);
      if (first_off_field && second_off_field) {
        continue;
      }
      est->pos1.invalid = first_off_field;
      est->pos2->invalid = second_off_field;
    } else if (first_off_field) {
      continue;
    }

    estimates.push_back(std::move(est.value()));
  }
  return estimates;
}

void UnambiguousSolverNode::Solve(
    const std::vector<std::vector<apriltag::tag_detection_t>>&
        detection_batches,
    bool reject_far_tags) {
  const std::optional<position_estimate_t> pose_estimate =
      SolveWithoutNotify(detection_batches, reject_far_tags);

  for (const auto& cb : callbacks_) {
    cb(pose_estimate);
  }
}

auto UnambiguousSolverNode::SolveWithoutNotify(
    const std::vector<std::vector<apriltag::tag_detection_t>>&
        detection_batches,
    bool reject_far_tags) -> std::optional<position_estimate_t> {
  const auto& ambiguous_estimates =
      GetAmbiguousEstimates(detection_batches, reject_far_tags);
  std::vector<position_estimate_t> best_solution;
  std::vector<position_estimate_t> current_solution;
  double best_cost = std::numeric_limits<double>::infinity();
  double cost = SearchSolutions(ambiguous_estimates, 0, current_solution,
                                best_solution, best_cost);

  if (best_solution.empty()) {
    return std::nullopt;
  }

  double avg_variance = 0;
  double avg_timestamp = 0;
  std::vector<int> tag_ids;
  for (const position_estimate_t& est : best_solution) {
    avg_variance += est.variance;
    avg_timestamp += est.timestamp;
    for (const int tag_id : est.tag_ids) {
      tag_ids.push_back(tag_id);
    }
  }
  avg_variance /= best_solution.size();
  avg_timestamp /= best_solution.size();

  position_estimate_t estimate{.tag_ids = tag_ids,
                               .pose = WeightedAveragePose(best_solution),
                               .variance = avg_variance,
                               .timestamp = avg_timestamp,
                               .num_tags = static_cast<int>(tag_ids.size()),
                               .loss = cost};

  prev_pose_estimate_ = estimate;
  return estimate;
}

}  // namespace localization
