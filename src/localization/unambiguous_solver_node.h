#pragma once
#include <functional>
#include <mutex>
#include <optional>
#include <vector>
#include "camera/camera_constants.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position.h"
#include "localization/position_solver.h"

namespace localization {

class UnambiguousSolverNode : public IAccumulatingSolverNode {
 public:
  UnambiguousSolverNode(
      const std::vector<camera::camera_constant_t>& camera_constants,
      const wpi::apriltag::AprilTagFieldLayout& layout = kapriltag_layout);

  void RegisterCallback(
      const std::function<
          void(position_estimate_t, control_loops::MetaDataList metadata,
               std::shared_ptr<control_loops::Context>)>& callback) override;

  void Accumulate(
      std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
      control_loops::MetaDataList metadata,
      std::shared_ptr<control_loops::Context> ctx) override;

  auto SolveWithoutNotify(
      const std::vector<std::vector<apriltag::tag_detection_t>>&
          detection_batches,
      bool reject_far_tags = true) -> std::optional<position_estimate_t>;
  auto SolveWithoutNotify(
      const std::vector<std::vector<apriltag::tag_detection_t>>&
          detection_batches,
      const std::vector<control_loops::MetaDataList>& metadata_batches,
      bool reject_far_tags = true) -> std::optional<position_estimate_t>;

 private:
  static auto Cost(const wpi::math::Pose3d& a,
                   const wpi::math::Pose3d& b) -> double;
  auto ComputeCost(const std::vector<position_estimate_t>& poses) -> double;
  static auto WeightedAveragePose(
      const std::vector<position_estimate_t>& solutions) -> wpi::math::Pose3d;
  auto SearchSolutions(
      const std::vector<ambiguous_estimate_t>& all_pose_estimates, size_t index,
      std::vector<position_estimate_t>& current_solution,
      std::vector<position_estimate_t>& best_solution,
      double& best_cost) -> double;
  auto GetAmbiguousEstimates(
      const std::vector<std::vector<apriltag::tag_detection_t>>&
          detection_batches,
      const std::vector<control_loops::MetaDataList>& metadata_batches,
      bool reject_far_tags) -> std::vector<ambiguous_estimate_t>;
  void SolveAndReset(std::unique_lock<std::mutex>& lock,
                     std::shared_ptr<control_loops::Context> ctx);

  int num_cameras_;
  std::vector<std::function<void(position_estimate_t,
                                 control_loops::MetaDataList metadata,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;
  std::vector<std::string> camera_names_;
  std::vector<MultiTagSolverNode> multitag_solvers_;

  std::mutex mutex_;
  int cameras_reported_{0};
  std::vector<bool> camera_reported_;
  std::vector<std::vector<apriltag::tag_detection_t>> accumulated_detections_;
  std::vector<control_loops::MetaDataList> accumulated_metadata_;

  std::optional<position_estimate_t> prev_pose_estimate_;
  static constexpr double kacceptable_frame_recency = 0.25;
};

}  // namespace localization
