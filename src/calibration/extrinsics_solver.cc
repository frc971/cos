#include "calibration/extrinsics_solver.h"

#include <array>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <unordered_set>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <nlohmann/json.hpp>

namespace calibration {
namespace {
constexpr double kTranslationResidualScaleMeters = 0.10;
constexpr double kRotationResidualScaleRadians = 10 * std::numbers::pi / 180;
constexpr double kRotationPriorScaleRadians = 5 * std::numbers::pi / 180;

auto ExtractFieldToCameraPose(const Observation& observation) -> ExtrinsicPose {
  const auto field_to_camera_matrix = observation.field_to_camera.ToMatrix();
  return {.rotation =
              Eigen::Quaterniond{field_to_camera_matrix.topLeftCorner<3, 3>()},
          .translation = field_to_camera_matrix.topRightCorner<3, 1>()};
}

template <typename T>
void WriteScalarFirstQuaternion(const Eigen::Quaterniond& rotation_quaternion,
                                std::array<T, 4>& scalar_first_quaternion) {
  scalar_first_quaternion[0] = T(rotation_quaternion.w());
  scalar_first_quaternion[1] = T(rotation_quaternion.x());
  scalar_first_quaternion[2] = T(rotation_quaternion.y());
  scalar_first_quaternion[3] = T(rotation_quaternion.z());
}

// Each camera parameter block stores camera_to_robot angle-axis rotation in
// slots 0..2 and translation in meters in slots 3..5.
template <typename T>
void ComposeFieldToRobotPose(const ExtrinsicPose& field_to_camera_pose,
                             const T* camera_to_robot_parameters,
                             std::array<T, 4>& field_to_robot_quaternion,
                             std::array<T, 3>& field_to_robot_translation) {
  std::array<T, 4> field_to_camera_quaternion;
  std::array<T, 4> camera_to_robot_quaternion;
  WriteScalarFirstQuaternion(field_to_camera_pose.rotation,
                             field_to_camera_quaternion);
  ceres::AngleAxisToQuaternion(camera_to_robot_parameters,
                               camera_to_robot_quaternion.data());
  ceres::QuaternionProduct(field_to_camera_quaternion.data(),
                           camera_to_robot_quaternion.data(),
                           field_to_robot_quaternion.data());
  ceres::QuaternionRotatePoint(field_to_camera_quaternion.data(),
                               camera_to_robot_parameters + 3,
                               field_to_robot_translation.data());
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    field_to_robot_translation[axis_index] +=
        T(field_to_camera_pose.translation[axis_index]);
  }
}

template <typename T>
void ComputeRelativeRotationAngleAxis(const T* reference_rotation_quaternion,
                                      const T* compared_rotation_quaternion,
                                      T* relative_rotation_angle_axis) {
  const std::array<T, 4> inverse_reference_rotation_quaternion = {
      reference_rotation_quaternion[0], -reference_rotation_quaternion[1],
      -reference_rotation_quaternion[2], -reference_rotation_quaternion[3]};
  std::array<T, 4> relative_rotation_quaternion;
  ceres::QuaternionProduct(inverse_reference_rotation_quaternion.data(),
                           compared_rotation_quaternion,
                           relative_rotation_quaternion.data());
  // Ceres uses the shortest rotation and has a finite derivative at identity.
  ceres::QuaternionToAngleAxis(relative_rotation_quaternion.data(),
                               relative_rotation_angle_axis);
}

struct CameraPairRobotPoseResidual {
  ExtrinsicPose camera_1_field_to_camera_pose;
  ExtrinsicPose camera_2_field_to_camera_pose;
  double pair_residual_normalization;

  template <typename T>
  auto operator()(const T* camera_1_camera_to_robot_parameters,
                  const T* camera_2_camera_to_robot_parameters,
                  T* normalized_robot_pose_residual) const -> bool {
    std::array<T, 4> camera_1_field_to_robot_quaternion;
    std::array<T, 4> camera_2_field_to_robot_quaternion;
    std::array<T, 3> camera_1_field_to_robot_translation;
    std::array<T, 3> camera_2_field_to_robot_translation;
    ComposeFieldToRobotPose(camera_1_field_to_camera_pose,
                            camera_1_camera_to_robot_parameters,
                            camera_1_field_to_robot_quaternion,
                            camera_1_field_to_robot_translation);
    ComposeFieldToRobotPose(camera_2_field_to_camera_pose,
                            camera_2_camera_to_robot_parameters,
                            camera_2_field_to_robot_quaternion,
                            camera_2_field_to_robot_translation);
    ComputeRelativeRotationAngleAxis(camera_1_field_to_robot_quaternion.data(),
                                     camera_2_field_to_robot_quaternion.data(),
                                     normalized_robot_pose_residual + 3);
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
      normalized_robot_pose_residual[axis_index] =
          (camera_1_field_to_robot_translation[axis_index] -
           camera_2_field_to_robot_translation[axis_index]) *
          T(pair_residual_normalization / kTranslationResidualScaleMeters);
      normalized_robot_pose_residual[axis_index + 3] *=
          T(pair_residual_normalization / kRotationResidualScaleRadians);
    }
    return true;
  }
};

struct CameraToRobotRotationPriorResidual {
  Eigen::Quaterniond initial_camera_to_robot_quaternion;

  template <typename T>
  auto operator()(const T* camera_to_robot_parameters,
                  T* normalized_rotation_prior_residual) const -> bool {
    std::array<T, 4> initial_camera_to_robot_scalar_first_quaternion;
    std::array<T, 4> optimized_camera_to_robot_quaternion;
    WriteScalarFirstQuaternion(initial_camera_to_robot_quaternion,
                               initial_camera_to_robot_scalar_first_quaternion);
    ceres::AngleAxisToQuaternion(camera_to_robot_parameters,
                                 optimized_camera_to_robot_quaternion.data());
    ComputeRelativeRotationAngleAxis(
        initial_camera_to_robot_scalar_first_quaternion.data(),
        optimized_camera_to_robot_quaternion.data(),
        normalized_rotation_prior_residual);
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
      normalized_rotation_prior_residual[axis_index] /=
          T(kRotationPriorScaleRadians);
    }
    return true;
  }
};

auto EncodeCameraToRobotParameters(const ExtrinsicPose& camera_to_robot_pose)
    -> std::array<double, 6> {
  std::array<double, 6> camera_to_robot_parameters{};
  std::array<double, 4> camera_to_robot_quaternion;
  WriteScalarFirstQuaternion(camera_to_robot_pose.rotation,
                             camera_to_robot_quaternion);
  ceres::QuaternionToAngleAxis(camera_to_robot_quaternion.data(),
                               camera_to_robot_parameters.data());
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    camera_to_robot_parameters[axis_index + 3] =
        camera_to_robot_pose.translation[axis_index];
  }
  return camera_to_robot_parameters;
}

void ValidateObservationsAndExtrinsics(
    const std::vector<ObservationGroup>& observation_groups,
    const std::vector<ExtrinsicPose>& camera_to_robot_extrinsics) {
  for (const auto& camera_to_robot_pose : camera_to_robot_extrinsics) {
    if (!camera_to_robot_pose.translation.allFinite() ||
        !camera_to_robot_pose.rotation.coeffs().allFinite() ||
        std::abs(camera_to_robot_pose.rotation.norm() - 1) > 1e-8) {
      throw std::invalid_argument(
          "Extrinsics must be finite with unit quaternions");
    }
  }
  for (const auto& observation_group : observation_groups) {
    if (observation_group.size() < 2) {
      throw std::invalid_argument("An observation group needs two cameras");
    }
    std::unordered_set<size_t> observed_camera_indices;
    for (const auto& observation : observation_group) {
      if (observation.camera >= camera_to_robot_extrinsics.size() ||
          !observed_camera_indices.insert(observation.camera).second ||
          !observation.field_to_camera.ToMatrix().allFinite() ||
          observation.capture_ns < 0) {
        throw std::invalid_argument("Invalid observation");
      }
    }
  }
}
}  // namespace

auto LoadExtrinsics(const std::vector<ReplayCamera>& cameras)
    -> std::vector<ExtrinsicPose> {
  std::vector<ExtrinsicPose> camera_to_robot_extrinsics;
  for (const auto& camera : cameras) {
    const auto camera_to_robot_matrix = camera::Extrinsics{camera.config_path}
                                            .ToCameraToRobot<frc::Transform3d>()
                                            .ToMatrix();
    camera_to_robot_extrinsics.push_back(
        {Eigen::Quaterniond{camera_to_robot_matrix.topLeftCorner<3, 3>()},
         camera_to_robot_matrix.topRightCorner<3, 1>()});
  }
  return camera_to_robot_extrinsics;
}

auto EvaluatePairErrors(
    const std::vector<ObservationGroup>& observation_groups,
    const std::vector<ExtrinsicPose>& camera_to_robot_extrinsics)
    -> PairErrors {
  ValidateObservationsAndExtrinsics(observation_groups,
                                    camera_to_robot_extrinsics);
  PairErrors pair_errors;
  double sum_squared_translation_errors_meters_squared = 0;
  double sum_squared_rotation_errors_radians_squared = 0;
  for (const auto& observation_group : observation_groups) {
    for (size_t camera_1_observation_index = 0;
         camera_1_observation_index < observation_group.size();
         ++camera_1_observation_index) {
      for (size_t camera_2_observation_index = camera_1_observation_index + 1;
           camera_2_observation_index < observation_group.size();
           ++camera_2_observation_index) {
        const auto& camera_1_observation =
            observation_group[camera_1_observation_index];
        const auto& camera_2_observation =
            observation_group[camera_2_observation_index];
        const auto camera_1_camera_to_robot_parameters =
            EncodeCameraToRobotParameters(
                camera_to_robot_extrinsics[camera_1_observation.camera]);
        const auto camera_2_camera_to_robot_parameters =
            EncodeCameraToRobotParameters(
                camera_to_robot_extrinsics[camera_2_observation.camera]);
        std::array<double, 6> normalized_robot_pose_residual;
        CameraPairRobotPoseResidual{
            ExtractFieldToCameraPose(camera_1_observation),
            ExtractFieldToCameraPose(camera_2_observation),
            1}(camera_1_camera_to_robot_parameters.data(),
               camera_2_camera_to_robot_parameters.data(),
               normalized_robot_pose_residual.data());
        for (int axis_index = 0; axis_index < 3; ++axis_index) {
          sum_squared_translation_errors_meters_squared +=
              std::pow(normalized_robot_pose_residual[axis_index] *
                           kTranslationResidualScaleMeters,
                       2);
          sum_squared_rotation_errors_radians_squared +=
              std::pow(normalized_robot_pose_residual[axis_index + 3] *
                           kRotationResidualScaleRadians,
                       2);
        }
        ++pair_errors.pairs;
      }
    }
  }
  if (pair_errors.pairs) {
    pair_errors.translation_rms_m = std::sqrt(
        sum_squared_translation_errors_meters_squared / pair_errors.pairs);
    pair_errors.rotation_rms_deg =
        std::sqrt(sum_squared_rotation_errors_radians_squared /
                  pair_errors.pairs) *
        180 / std::numbers::pi;
    pair_errors.normalized_half_mean_squared =
        0.5 / pair_errors.pairs *
        (sum_squared_translation_errors_meters_squared /
             std::pow(kTranslationResidualScaleMeters, 2) +
         sum_squared_rotation_errors_radians_squared /
             std::pow(kRotationResidualScaleRadians, 2));
  }
  return pair_errors;
}

auto SolveExtrinsics(
    const std::vector<ObservationGroup>& observation_groups,
    const std::vector<ExtrinsicPose>& initial_camera_to_robot_extrinsics,
    size_t anchor_camera_index) -> CalibrationResult {
  if (anchor_camera_index >= initial_camera_to_robot_extrinsics.size()) {
    throw std::invalid_argument("Invalid anchor camera");
  }
  CalibrationResult calibration_result;
  calibration_result.initial_errors = EvaluatePairErrors(
      observation_groups, initial_camera_to_robot_extrinsics);
  if (!calibration_result.initial_errors.pairs) {
    throw std::invalid_argument("No training pairs");
  }
  // Every optimized camera needs a path to the anchor in the observation graph.
  std::vector<bool> camera_connected_to_anchor(
      initial_camera_to_robot_extrinsics.size(), false);
  camera_connected_to_anchor[anchor_camera_index] = true;
  for (size_t connectivity_pass = 0;
       connectivity_pass < initial_camera_to_robot_extrinsics.size();
       ++connectivity_pass) {
    for (const auto& observation_group : observation_groups) {
      bool group_connected_to_anchor = false;
      for (const auto& observation : observation_group) {
        group_connected_to_anchor |=
            camera_connected_to_anchor[observation.camera];
      }
      if (group_connected_to_anchor) {
        for (const auto& observation : observation_group) {
          camera_connected_to_anchor[observation.camera] = true;
        }
      }
    }
  }
  for (bool connected_to_anchor : camera_connected_to_anchor) {
    if (!connected_to_anchor) {
      throw std::invalid_argument(
          "Training cameras are not connected to the anchor");
    }
  }

  std::vector<std::array<double, 6>> camera_to_robot_parameter_blocks;
  for (const auto& camera_to_robot_pose : initial_camera_to_robot_extrinsics) {
    camera_to_robot_parameter_blocks.push_back(
        EncodeCameraToRobotParameters(camera_to_robot_pose));
  }
  ceres::Problem calibration_problem;
  const double pair_residual_normalization =
      1 /
      std::sqrt(static_cast<double>(calibration_result.initial_errors.pairs));
  for (const auto& observation_group : observation_groups) {
    for (size_t camera_1_observation_index = 0;
         camera_1_observation_index < observation_group.size();
         ++camera_1_observation_index) {
      for (size_t camera_2_observation_index = camera_1_observation_index + 1;
           camera_2_observation_index < observation_group.size();
           ++camera_2_observation_index) {
        const auto& camera_1_observation =
            observation_group[camera_1_observation_index];
        const auto& camera_2_observation =
            observation_group[camera_2_observation_index];
        calibration_problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CameraPairRobotPoseResidual, 6, 6,
                                            6>(new CameraPairRobotPoseResidual{
                ExtractFieldToCameraPose(camera_1_observation),
                ExtractFieldToCameraPose(camera_2_observation),
                pair_residual_normalization}),
            nullptr,
            camera_to_robot_parameter_blocks[camera_1_observation.camera]
                .data(),
            camera_to_robot_parameter_blocks[camera_2_observation.camera]
                .data());
      }
    }
  }
  for (size_t camera_index = 0;
       camera_index < initial_camera_to_robot_extrinsics.size();
       ++camera_index) {
    if (camera_index == anchor_camera_index) {
      calibration_problem.SetParameterBlockConstant(
          camera_to_robot_parameter_blocks[camera_index].data());
    } else {
      calibration_problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<CameraToRobotRotationPriorResidual, 3,
                                          6>(
              new CameraToRobotRotationPriorResidual{
                  initial_camera_to_robot_extrinsics[camera_index].rotation}),
          nullptr, camera_to_robot_parameter_blocks[camera_index].data());
    }
  }
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = 200;
  solver_options.function_tolerance = 1e-12;
  solver_options.gradient_tolerance = 1e-12;
  solver_options.parameter_tolerance = 1e-12;
  ceres::Solver::Summary solver_summary;
  ceres::Solve(solver_options, &calibration_problem, &solver_summary);
  calibration_result.usable = solver_summary.IsSolutionUsable();
  calibration_result.solver_report = solver_summary.BriefReport();
  for (size_t camera_index = 0;
       camera_index < initial_camera_to_robot_extrinsics.size();
       ++camera_index) {
    if (camera_index == anchor_camera_index) {
      // Preserve the anchor exactly.
      calibration_result.extrinsics.push_back(
          initial_camera_to_robot_extrinsics[camera_index]);
    } else {
      const auto& optimized_camera_to_robot_parameters =
          camera_to_robot_parameter_blocks[camera_index];
      std::array<double, 4> optimized_camera_to_robot_quaternion;
      ceres::AngleAxisToQuaternion(optimized_camera_to_robot_parameters.data(),
                                   optimized_camera_to_robot_quaternion.data());
      calibration_result.extrinsics.push_back(
          {Eigen::Quaterniond{optimized_camera_to_robot_quaternion[0],
                              optimized_camera_to_robot_quaternion[1],
                              optimized_camera_to_robot_quaternion[2],
                              optimized_camera_to_robot_quaternion[3]},
           Eigen::Vector3d{optimized_camera_to_robot_parameters[3],
                           optimized_camera_to_robot_parameters[4],
                           optimized_camera_to_robot_parameters[5]}});
    }
  }
  calibration_result.final_errors =
      EvaluatePairErrors(observation_groups, calibration_result.extrinsics);
  return calibration_result;
}

void WriteCandidateConfigs(
    const std::vector<ReplayCamera>& cameras,
    const std::vector<ExtrinsicPose>& camera_to_robot_extrinsics,
    const std::filesystem::path& output_directory) {
  if (cameras.size() != camera_to_robot_extrinsics.size()) {
    throw std::invalid_argument("Camera count mismatch");
  }
  ValidateObservationsAndExtrinsics({}, camera_to_robot_extrinsics);
  std::unordered_set<std::string> candidate_config_filenames;
  std::vector<nlohmann::json> candidate_camera_configs;
  for (size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
    const auto config_filename = cameras[camera_index].config_path.filename();
    if (!candidate_config_filenames.insert(config_filename.string()).second ||
        std::filesystem::exists(output_directory / config_filename)) {
      throw std::invalid_argument(
          "Refusing to overwrite candidate configuration: " +
          config_filename.string());
    }
    std::ifstream camera_config_input(cameras[camera_index].config_path);
    auto camera_config_json = nlohmann::json::parse(camera_config_input);
    // JSON stores the inverse pose: camera position and Euler angles in robot.
    const auto& camera_to_robot_pose = camera_to_robot_extrinsics[camera_index];
    const Eigen::Vector3d robot_to_camera_translation =
        -(camera_to_robot_pose.rotation.conjugate() *
          camera_to_robot_pose.translation);
    const auto robot_to_camera_quaternion =
        camera_to_robot_pose.rotation.conjugate();
    const frc::Rotation3d robot_to_camera_rotation{frc::Quaternion{
        robot_to_camera_quaternion.w(), robot_to_camera_quaternion.x(),
        robot_to_camera_quaternion.y(), robot_to_camera_quaternion.z()}};
    const auto original_robot_to_camera_matrix =
        camera::Extrinsics{cameras[camera_index].config_path}
            .ToRobotToCamera<frc::Transform3d>()
            .ToMatrix();
    const frc::Pose3d candidate_robot_to_camera_pose{
        units::meter_t{robot_to_camera_translation.x()},
        units::meter_t{robot_to_camera_translation.y()},
        units::meter_t{robot_to_camera_translation.z()},
        robot_to_camera_rotation};
    // Preserve the anchor's original JSON numbers as well as its transform.
    if (!candidate_robot_to_camera_pose.ToMatrix().isApprox(
            original_robot_to_camera_matrix, 1e-12)) {
      auto& extrinsics_json = camera_config_json["extrinsics"];
      extrinsics_json["translation_x"] = robot_to_camera_translation.x();
      extrinsics_json["translation_y"] = robot_to_camera_translation.y();
      extrinsics_json["translation_z"] = robot_to_camera_translation.z();
      extrinsics_json["rotation_x"] =
          units::degree_t{robot_to_camera_rotation.X()}.value();
      extrinsics_json["rotation_y"] =
          units::degree_t{robot_to_camera_rotation.Y()}.value();
      extrinsics_json["rotation_z"] =
          units::degree_t{robot_to_camera_rotation.Z()}.value();
    }
    candidate_camera_configs.push_back(std::move(camera_config_json));
  }
  std::filesystem::create_directories(output_directory);
  for (size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
    std::ofstream candidate_config_output(
        output_directory / cameras[camera_index].config_path.filename());
    candidate_config_output << candidate_camera_configs[camera_index].dump(2)
                            << '\n';
    if (!candidate_config_output) {
      throw std::runtime_error("Cannot write candidate configuration");
    }
  }
}

}  // namespace calibration
