#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "tools/charuco_calibration.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, detections_path, "",  // NOLINT
          "ChArUco detection JSON exported by calib_helper; calibrates "
          "without loading images");  // NOLINT
ABSL_FLAG(int, max_detections, 50,    // NOLINT
          "calibration captures selected in 0.5-second steps "
          "(0 uses all usable detections)");          // NOLINT
ABSL_FLAG(std::string, intrinsics_output_path,        // NOLINT
          "intrinsics.json",                          // NOLINT
          "path for the generated intrinsics JSON");  // NOLINT

namespace {

using json = nlohmann::json;
using charuco_calibration::CalibrateCamera;
using charuco_calibration::DetectionFromJson;
using charuco_calibration::DetectionResult;
using charuco_calibration::HasEnoughCorners;
using charuco_calibration::IntrinsicsToJson;

auto WriteIntrinsicsToFile(const cv::Mat& camera_matrix,
                           const cv::Mat& dist_coeffs, const std::string& path)
    -> void {
  std::ofstream intrinsics_file(path);
  CHECK(intrinsics_file.is_open()) << "Failed to open " << path;
  const json intrinsics = IntrinsicsToJson(camera_matrix, dist_coeffs);
  intrinsics_file << intrinsics.dump(4) << '\n';

  LOG(INFO) << "Intrinsics:\n" << intrinsics.dump(4);
}

auto RunCalibration(const std::vector<DetectionResult>& detection_results,
                    cv::Size image_size) -> int {
  LOG(INFO) << "Calibrating with " << detection_results.size()
            << " captured frames";

  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  std::optional<double> reprojection_error = CalibrateCamera(
      detection_results, image_size, &camera_matrix, &dist_coeffs);
  if (!reprojection_error.has_value()) {
    LOG(ERROR) << "No usable detections captured";
    return 1;
  }

  LOG(INFO) << "Reprojection error: " << *reprojection_error;
  WriteIntrinsicsToFile(camera_matrix, dist_coeffs,
                        absl::GetFlag(FLAGS_intrinsics_output_path));
  return 0;
}

auto CalibrateDetectionsFile(const std::string& path, int max_detections)
    -> int {
  try {
    std::ifstream input(path);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open " + path);
    }
    json saved;
    input >> saved;
    const cv::Size image_size(saved.at("image_size").at("width").get<int>(),
                              saved.at("image_size").at("height").get<int>());
    const auto& detections = saved.at("detections");
    const auto limit = static_cast<std::size_t>(max_detections);
    std::vector<DetectionResult> results;
    if (max_detections == 0) {
      for (const auto& detection : detections) {
        if (stop::stop) {
          return 0;
        }
        auto result = DetectionFromJson(detection);
        if (HasEnoughCorners(result)) {
          results.push_back(std::move(result));
        }
      }
    } else {
      std::vector<std::pair<double, std::size_t>> frames;
      for (std::size_t index = 0; index < detections.size(); ++index) {
        const std::filesystem::path filename =
            detections[index].at("filename").get<std::string>();
        frames.emplace_back(std::stod(filename.stem().string()), index);
      }
      std::ranges::sort(frames);
      std::vector<bool> selected(detections.size(), false);
      for (const double offset : {0.0, 0.25}) {
        if (frames.empty() || results.size() == limit) {
          break;
        }
        double next_timestamp = frames.front().first + offset;
        for (const auto& [timestamp, index] : frames) {
          if (stop::stop) {
            return 0;
          }
          if (timestamp < next_timestamp) {
            continue;
          }
          next_timestamp = timestamp + 0.5;
          if (selected[index]) {
            continue;
          }
          auto result = DetectionFromJson(detections[index]);
          if (HasEnoughCorners(result)) {
            selected[index] = true;
            results.push_back(std::move(result));
            LOG(INFO) << "Selected frame: "
                      << detections[index].at("filename").get<std::string>();
            if (results.size() == limit) {
              break;
            }
          }
        }
      }
      if (results.size() < limit) {
        throw std::runtime_error("Only " + std::to_string(results.size()) +
                                 " usable frames after both sampling passes; " +
                                 std::to_string(max_detections) + " required");
      }
    }
    LOG(INFO) << "Selected " << results.size() << " of " << detections.size()
              << " saved detections from " << path;
    return RunCalibration(results, image_size);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to calibrate saved detections: " << error.what();
    return 1;
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  stop::RegisterHandler();

  const std::string detections_path = absl::GetFlag(FLAGS_detections_path);
  CHECK(!detections_path.empty()) << "--detections_path is required";
  const int max_detections = absl::GetFlag(FLAGS_max_detections);
  CHECK_GE(max_detections, 0) << "--max_detections must be nonnegative";

  return CalibrateDetectionsFile(detections_path, max_detections);
}
