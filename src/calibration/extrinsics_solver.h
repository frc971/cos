#pragma once

#include <Eigen/Geometry>
#include <string>
#include <vector>

#include "calibration/extrinsics_replay.h"

namespace calibration {

struct ExtrinsicPose {
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

struct PairErrors {
  size_t pairs = 0;
  double translation_rms_m = 0;
  double rotation_rms_deg = 0;
  double normalized_half_mean_squared = 0;
};

struct CalibrationResult {
  std::vector<ExtrinsicPose> extrinsics;  // T_C<-R, in WPILib coordinates.
  PairErrors initial_errors;
  PairErrors final_errors;
  bool usable = false;
  std::string solver_report;
};

auto LoadExtrinsics(const std::vector<ReplayCamera>& cameras)
    -> std::vector<ExtrinsicPose>;
// Ordinary squared pair loss divided by the number of pairs, plus one
// squared SO(3) correction penalty per optimized camera at a 5 degree scale.
auto SolveExtrinsics(
    const std::vector<ObservationGroup>& observation_groups,
    const std::vector<ExtrinsicPose>& initial_camera_to_robot_extrinsics,
    size_t anchor_camera_index) -> CalibrationResult;
auto EvaluatePairErrors(
    const std::vector<ObservationGroup>& observation_groups,
    const std::vector<ExtrinsicPose>& camera_to_robot_extrinsics) -> PairErrors;
// Convert back to the JSON T_R<-C convention (meters and Euler degrees),
// retaining every other input configuration field. Existing files are errors.
void WriteCandidateConfigs(
    const std::vector<ReplayCamera>& cameras,
    const std::vector<ExtrinsicPose>& camera_to_robot_extrinsics,
    const std::filesystem::path& output_directory);

}  // namespace calibration
