#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/charuco_calibration.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, detections_path, "",  // NOLINT
          "ChArUco detection JSON exported by calib_helper; calibrates "
          "without loading images");  // NOLINT
ABSL_FLAG(int, max_detections, 50,  // NOLINT
          "calibration captures sampled evenly across 20 radius bands and "
          "20 angle sectors relative to the image center "
          "(0 uses all usable detections)");  // NOLINT
ABSL_FLAG(std::string, intrinsics_output_path,        // NOLINT
          "intrinsics.json",                          // NOLINT
          "path for the generated intrinsics JSON");  // NOLINT
ABSL_FLAG(std::string, coverage_output_path,             // NOLINT
          "calibration_coverage.png",                    // NOLINT
          "path for the detections and density image");  // NOLINT

namespace {

using json = nlohmann::json;
using charuco_calibration::CalibrateCamera;
using charuco_calibration::DetectionFromJson;
using charuco_calibration::DetectionResult;
using charuco_calibration::HasEnoughCorners;
using charuco_calibration::IntrinsicsToJson;

constexpr std::size_t kBlocksPerTrait = 20;

auto WriteCoverageImage(const std::vector<DetectionResult>& detections,
                        cv::Size image_size) -> void {
  cv::Mat overlay = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat density = cv::Mat::zeros(image_size, CV_32FC1);
  for (const auto& detection : detections) {
    cv::aruco::drawDetectedCornersCharuco(
        overlay, detection.charuco_corners, detection.charuco_ids,
        cv::Scalar(255, 0, 0));
    for (const auto& point : detection.image_points) {
      if (std::isfinite(point.x) && std::isfinite(point.y) && point.x >= 0 &&
          point.y >= 0 && point.x < image_size.width &&
          point.y < image_size.height) {
        density.at<float>(static_cast<int>(point.y),
                          static_cast<int>(point.x)) += 1.0F;
      }
    }
  }
  // Accumulate before smoothing so overlapping captures increase density.
  const double sigma = std::max(1.0, image_size.width / 100.0);
  cv::GaussianBlur(density, density, cv::Size(), sigma, sigma);
  cv::Mat normalized;
  cv::normalize(density, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::Mat heatmap;
  cv::applyColorMap(normalized, heatmap, cv::COLORMAP_TURBO);
  cv::Mat frame;
  cv::hconcat(overlay, heatmap, frame);
  cv::putText(frame, "Selected detections: " + std::to_string(detections.size()),
              cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2);
  cv::putText(frame, "Corner density: blue = low, red = high",
              cv::Point(image_size.width + 20, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.7, cv::Scalar(255, 255, 255), 2);
  const auto path = absl::GetFlag(FLAGS_coverage_output_path);
  if (!cv::imwrite(path, frame)) {
    throw std::runtime_error("Failed to write " + path);
  }
  LOG(INFO) << "Wrote calibration coverage to " << path;
}

auto DetectionStratum(const DetectionResult& detection, cv::Size image_size)
    -> std::size_t {
  cv::Point2d centroid;
  for (const auto& point : detection.image_points) {
    centroid.x += point.x;
    centroid.y += point.y;
  }
  centroid /= static_cast<double>(detection.image_points.size());
  const double x = centroid.x - image_size.width / 2.0;
  const double y = centroid.y - image_size.height / 2.0;
  const double max_radius =
      std::hypot(image_size.width / 2.0, image_size.height / 2.0);
  const double radius = std::hypot(x, y) / max_radius;
  const double angle =
      (std::atan2(y, x) + std::numbers::pi) / (2.0 * std::numbers::pi);
  if (!std::isfinite(radius) || !std::isfinite(angle)) {
    throw std::runtime_error("Detection position must be finite");
  }
  const auto block = [](double position) -> std::size_t {
    return std::min(kBlocksPerTrait - 1,
                    static_cast<std::size_t>(
                        std::clamp(position, 0.0, 1.0) * kBlocksPerTrait));
  };
  return block(radius) * kBlocksPerTrait + block(angle);
}

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
    if (image_size.width <= 0 || image_size.height <= 0) {
      throw std::runtime_error("Image dimensions must be positive");
    }
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
      std::vector<std::pair<DetectionResult, std::size_t>> candidates;
      std::array<std::vector<std::size_t>, kBlocksPerTrait * kBlocksPerTrait>
          strata;
      for (std::size_t index = 0; index < detections.size(); ++index) {
        if (stop::stop) {
          return 0;
        }
        auto result = DetectionFromJson(detections[index]);
        if (HasEnoughCorners(result)) {
          strata[DetectionStratum(result, image_size)].push_back(
              candidates.size());
          candidates.emplace_back(std::move(result), index);
        }
      }
      if (candidates.size() < limit) {
        throw std::runtime_error("Only " + std::to_string(candidates.size()) +
                                 " usable frames; " +
                                 std::to_string(max_detections) + " required");
      }
      std::mt19937 generator(std::random_device{}());
      std::vector<std::size_t> populated_strata;
      for (std::size_t index = 0; index < strata.size(); ++index) {
        if (!strata[index].empty()) {
          std::shuffle(strata[index].begin(), strata[index].end(), generator);
          populated_strata.push_back(index);
        }
      }
      // Draw once per populated stratum per round. Randomize the order so a
      // partial final round does not favor smaller radii or angles.
      while (results.size() < limit) {
        std::shuffle(populated_strata.begin(), populated_strata.end(),
                     generator);
        for (const auto stratum : populated_strata) {
          if (stop::stop) {
            return 0;
          }
          auto& bucket = strata[stratum];
          auto& [result, index] = candidates[bucket.back()];
          bucket.pop_back();
          results.push_back(std::move(result));
          LOG(INFO) << "Selected frame: "
                    << detections[index].at("filename").get<std::string>();
          if (results.size() == limit) {
            break;
          }
        }
        std::erase_if(populated_strata, [&strata](std::size_t index) -> bool {
          return strata[index].empty();
        });
      }
    }
    LOG(INFO) << "Selected " << results.size() << " of " << detections.size()
              << " saved detections from " << path;
    if (!results.empty()) {
      WriteCoverageImage(results, image_size);
    }
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
