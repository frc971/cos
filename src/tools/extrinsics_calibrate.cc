#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "calibration/extrinsics_solver.h"

ABSL_FLAG(std::string, replay_directory, "",
          "Directory containing camera JPEG logs");
ABSL_FLAG(std::string, camera_config_directory, "/root/constants/second_bot",
          "Input camera JSON directory");
ABSL_FLAG(std::string, output_directory, "",
          "Empty directory for candidate JSONs and reports");
ABSL_FLAG(std::vector<std::string>, camera_names,
          (std::vector<std::string>{"second_bot_front", "second_bot_left",
                                    "second_bot_right"}),
          "Comma-separated camera identities and replay subdirectory names");
ABSL_FLAG(std::vector<std::string>, config_files,
          (std::vector<std::string>{"front_camera.json", "left_camera.json",
                                    "right_camera.json"}),
          "Comma-separated JSON filenames, in camera_names order");
ABSL_FLAG(std::string, anchor_camera, "second_bot_front",
          "Camera extrinsic to hold fixed");
ABSL_FLAG(bool, reject_far_tags, true,
          "Apply existing localization sanity checks");

namespace {
auto Errors(const calibration::PairErrors& e) -> nlohmann::json {
  return {{"pairs", e.pairs},
          {"translation_rms_m", e.translation_rms_m},
          {"rotation_rms_deg", e.rotation_rms_deg},
          {"normalized_half_mean_squared", e.normalized_half_mean_squared}};
}
}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kWarning);
  try {
    const std::filesystem::path replay = absl::GetFlag(FLAGS_replay_directory);
    const std::filesystem::path output = absl::GetFlag(FLAGS_output_directory);
    const std::filesystem::path configs =
        absl::GetFlag(FLAGS_camera_config_directory);
    if (replay.empty() || output.empty())
      throw std::invalid_argument(
          "--replay_directory and --output_directory are required");
    if (std::filesystem::exists(output) &&
        (!std::filesystem::is_directory(output) ||
         !std::filesystem::is_empty(output)))
      throw std::invalid_argument(
          "Output directory must be empty; inputs will not be overwritten");
    const auto names = absl::GetFlag(FLAGS_camera_names);
    const auto files = absl::GetFlag(FLAGS_config_files);
    if (names.size() < 2 || names.size() != files.size())
      throw std::invalid_argument(
          "Specify at least two camera names and equally many config files");
    std::vector<calibration::ReplayCamera> cameras;
    size_t anchor = names.size();
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == absl::GetFlag(FLAGS_anchor_camera))
        anchor = i;
      auto log = replay / names[i];
      // Also accept the front/left/right layout used by existing replay tools.
      if (!std::filesystem::exists(log) && names[i].starts_with("second_bot_"))
        log = replay / names[i].substr(std::string("second_bot_").size());
      if (!std::filesystem::is_regular_file(configs / files[i]))
        throw std::invalid_argument("Missing camera configuration: " +
                                    (configs / files[i]).string());
      cameras.push_back({.name = names[i],
                         .config_path = configs / files[i],
                         .log_directory = log});
    }
    if (anchor == names.size())
      throw std::invalid_argument("Anchor camera is not in camera_names");
    const auto initial = calibration::LoadExtrinsics(cameras);
    std::cout << "Replaying synchronized camera batches..." << std::endl;
    const auto replay_result = calibration::ReplayObservations(
        cameras, absl::GetFlag(FLAGS_reject_far_tags));
    const auto& groups = replay_result.groups;
    const auto split = calibration::SplitGroups(groups);
    const auto result =
        calibration::SolveExtrinsics(split.training, initial, anchor);
    if (!result.usable)
      throw std::runtime_error(result.solver_report);
    nlohmann::json report = {
        {"jpegs_processed", replay_result.jpegs_processed},
        {"valid_selected_pnp_observations", replay_result.observations.size()},
        {"observation_groups", groups.size()},
        {"observation_pairs",
         calibration::EvaluatePairErrors(groups, initial).pairs},
        {"training_groups", split.training.size()},
        {"held_out_groups", split.held_out.size()},
        {"anchor_camera", names[anchor]},
        {"training_initial", Errors(result.initial_errors)},
        {"training_final", Errors(result.final_errors)},
        {"held_out_initial",
         Errors(calibration::EvaluatePairErrors(split.held_out, initial))},
        {"held_out_final", Errors(calibration::EvaluatePairErrors(
                               split.held_out, result.extrinsics))},
        {"solver", result.solver_report},
        {"camera_changes", nlohmann::json::array()}};
    for (size_t i = 0; i < cameras.size(); ++i) {
      const auto& a = initial[i];
      const auto& b = result.extrinsics[i];
      const Eigen::Vector3d delta = -(b.rotation.conjugate() * b.translation) +
                                    a.rotation.conjugate() * a.translation;
      report["camera_changes"].push_back(
          {{"camera", names[i]},
           {"translation_change_m", {delta.x(), delta.y(), delta.z()}},
           {"translation_change_norm_m", delta.norm()},
           {"rotation_change_deg",
            a.rotation.angularDistance(b.rotation) * 180 / std::numbers::pi}});
    }
    calibration::WriteCandidateConfigs(cameras, result.extrinsics, output);
    std::ofstream report_file(output / "report.json");
    report_file << report.dump(2) << '\n';
    if (!report_file)
      throw std::runtime_error("Cannot write report");
    std::ofstream observations(output / "observations.csv");
    observations.precision(17);
    observations << "frame_id,camera,capture_ns,x,y,z,qw,qx,qy,qz,robot_x,"
                    "robot_y,robot_z,robot_qw,robot_qx,robot_qy,robot_qz\n";
    for (const auto& o : replay_result.observations) {
      const auto q = o.field_to_camera.Rotation().GetQuaternion();
      const auto r = o.field_to_robot.Rotation().GetQuaternion();
      observations << o.frame_id << ',' << cameras[o.camera].name << ','
                   << o.capture_ns << ',' << o.field_to_camera.X().value()
                   << ',' << o.field_to_camera.Y().value() << ','
                   << o.field_to_camera.Z().value() << ',' << q.W() << ','
                   << q.X() << ',' << q.Y() << ',' << q.Z() << ','
                   << o.field_to_robot.X().value() << ','
                   << o.field_to_robot.Y().value() << ','
                   << o.field_to_robot.Z().value() << ',' << r.W() << ','
                   << r.X() << ',' << r.Y() << ',' << r.Z() << '\n';
    }
    if (!observations)
      throw std::runtime_error("Cannot write observations");
    std::cout << report.dump(2) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "Calibration failed: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
