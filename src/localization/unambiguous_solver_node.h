#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <frc/apriltag/AprilTagFieldLayout.h>
#include "control_loop/node.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position.h"
#include "localization/solver_common.h"

namespace localization {

class UnambiguousSolverNode final : public control_loop::INode {
 public:
  UnambiguousSolverNode(std::string_view output_channel,
                        frc::AprilTagFieldLayout layout = kApriltagLayout);

  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void SetRejectFarTags(bool reject_far_tags);

  auto Solve(const std::vector<ambiguous_estimate_t*>& estimates,
             bool reject_far_tags = true) -> std::optional<position_estimate_t>;
  // Register the JPEG channel used to timestamp the fused pose.
  void AddCameraTimestamp(std::string_view jpeg_channel);
  void AddCamera(std::string_view input_channel,
                 const camera::Intrinsics& intrinsics,
                 const camera::Extrinsics& extrinsics,
                 control_loop::ControlLoop& control_loop);

 private:
  static auto Cost(const frc::Pose3d& a, const frc::Pose3d& b) -> double;
  auto ComputeCost(const std::vector<solver_estimate_t>& poses) -> double;
  static auto WeightedAveragePose(
      const std::vector<solver_estimate_t>& solutions) -> frc::Pose3d;
  auto SearchSolutions(
      const std::vector<ambiguous_estimate_t*>& all_pose_estimates,
      size_t index, std::vector<solver_estimate_t>& current_solution,
      std::vector<solver_estimate_t>& best_solution, double& best_cost)
      -> double;
  auto GetAmbiguousEstimates(
      const std::vector<std::vector<tag_detection_t>>& detection_batches,
      bool reject_far_tags) -> std::vector<ambiguous_estimate_t>;

  std::string output_channel_;
  std::vector<std::string> camera_timestamp_channels_;
  frc::AprilTagFieldLayout layout_;
  std::vector<std::string> detection_batch_channels_;
  struct PendingDetectionBatch {
    std::weak_ptr<control_loop::ContextInternal> context;
    size_t count = 0;
  };
  std::mutex solve_mutex_;
  std::unordered_map<const control_loop::ContextInternal*,
                     PendingDetectionBatch>
      ready_detection_batches_;
  std::vector<std::shared_ptr<MultiTagSolverNode>> multitag_solvers_;
  std::vector<std::string> multi_tag_solver_output_channels_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::optional<position_estimate_t> prev_pose_estimate_;
  bool reject_far_tags_ = true;
};

}  // namespace localization
