#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

namespace charuco_calibration {

constexpr static int ksquares_x = 12;
constexpr static int ksquares_y = 9;
constexpr static float ksquares_length = 0.06;
constexpr static float kpixel_per_square = 128;
constexpr static float kmarker_length = 0.045;
constexpr static int kmargin_squares = 0;

struct DetectionResult {
  cv::Mat charuco_corners;
  cv::Mat charuco_ids;
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  std::vector<int> marker_ids;
};

inline auto CreateBoard() -> cv::aruco::CharucoBoard {
  return {cv::Size(ksquares_x, ksquares_y), ksquares_length, kmarker_length,
          cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250)};
}

inline auto HasEnoughCorners(const DetectionResult& result) -> bool {
  if (result.charuco_corners.total() <= 3U || result.image_points.empty() ||
      result.object_points.empty()) {
    return false;
  }
  // Calibration initializes each planar view with a homography, which requires
  // noncollinear board corners even when more than three corners were detected.
  static const auto board = CreateBoard();
  return !board.checkCharucoCornersCollinear(result.charuco_ids);
}

inline auto CreateDetector(const cv::aruco::CharucoBoard& board)
    -> cv::aruco::CharucoDetector {
  cv::aruco::CharucoParameters charuco_params;
  cv::aruco::DetectorParameters detector_params;
  return {board, charuco_params, detector_params};
}

inline auto GenerateBoardImage(const cv::aruco::CharucoBoard& board)
    -> cv::Mat {
  const cv::Size image_size(
      static_cast<int>((ksquares_x + 2 * kmargin_squares) * kpixel_per_square),
      static_cast<int>((ksquares_y + 2 * kmargin_squares) * kpixel_per_square));

  cv::Mat board_image;
  board.generateImage(image_size, board_image,
                      static_cast<int>(kmargin_squares * kpixel_per_square), 1);
  return board_image;
}

inline auto DetectCharucoBoard(const cv::Mat& frame,
                               const cv::aruco::CharucoDetector& detector)
    -> DetectionResult {
  DetectionResult result;
  detector.detectBoard(frame, result.charuco_corners, result.charuco_ids,
                       result.marker_corners, result.marker_ids);
  if (result.charuco_corners.total() > 3U) {
    detector.getBoard().matchImagePoints(
        result.charuco_corners, result.charuco_ids, result.object_points,
        result.image_points);
  }
  return result;
}

inline auto CalibrateCamera(
    const std::vector<DetectionResult>& detection_results, cv::Size image_size,
    cv::Mat* camera_matrix, cv::Mat* dist_coeffs) -> std::optional<double> {
  std::vector<std::vector<cv::Point2f>> all_image_points;
  std::vector<std::vector<cv::Point3f>> all_object_points;

  for (const DetectionResult& detection_result : detection_results) {
    if (HasEnoughCorners(detection_result)) {
      all_image_points.push_back(detection_result.image_points);
      all_object_points.push_back(detection_result.object_points);
    }
  }

  if (all_image_points.empty()) {
    return std::nullopt;
  }

  return cv::calibrateCamera(all_object_points, all_image_points, image_size,
                             *camera_matrix, *dist_coeffs, cv::noArray(),
                             cv::noArray(), cv::noArray(), cv::noArray(),
                             cv::noArray());
}

inline auto IntrinsicsToJson(const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs) -> nlohmann::json {
  if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
    throw std::runtime_error("Camera matrix must be 3 by 3");
  }

  cv::Mat coeffs = dist_coeffs.reshape(1, 1);
  auto coeff = [&coeffs](int index) -> double {
    if (std::cmp_greater_equal(index, coeffs.total())) {
      return 0.0;
    }
    return coeffs.at<double>(0, index);
  };

  nlohmann::json output;
  output["fx"] = camera_matrix.at<double>(0, 0);
  output["cx"] = camera_matrix.at<double>(0, 2);
  output["fy"] = camera_matrix.at<double>(1, 1);
  output["cy"] = camera_matrix.at<double>(1, 2);
  output["k1"] = coeff(0);
  output["k2"] = coeff(1);
  output["p1"] = coeff(2);
  output["p2"] = coeff(3);
  output["k3"] = coeff(4);
  return output;
}

// Store matrices as arrays of coordinates/IDs rather than OpenCV metadata.
inline auto DetectionToJson(const DetectionResult& result) -> nlohmann::json {
  nlohmann::json corners = nlohmann::json::array();
  nlohmann::json ids = nlohmann::json::array();
  nlohmann::json image_points = nlohmann::json::array();
  nlohmann::json object_points = nlohmann::json::array();
  for (std::size_t i = 0; i < result.charuco_corners.total(); ++i) {
    const auto& corner =
        result.charuco_corners.at<cv::Point2f>(static_cast<int>(i));
    corners.push_back({corner.x, corner.y});
    ids.push_back(result.charuco_ids.at<int>(static_cast<int>(i)));
  }
  for (const auto& point : result.image_points) {
    image_points.push_back({point.x, point.y});
  }
  for (const auto& point : result.object_points) {
    object_points.push_back({point.x, point.y, point.z});
  }
  return {{"charuco_corners", corners},
          {"charuco_ids", ids},
          {"image_points", image_points},
          {"object_points", object_points}};
}

inline auto DetectionFromJson(const nlohmann::json& saved) -> DetectionResult {
  const auto& corners = saved.at("charuco_corners");
  const auto& ids = saved.at("charuco_ids");
  const auto& image_points = saved.at("image_points");
  const auto& object_points = saved.at("object_points");
  DetectionResult result;
  for (std::size_t i = 0; i < corners.size(); ++i) {
    result.charuco_corners.push_back(cv::Point2f(
        corners[i].at(0).get<float>(), corners[i].at(1).get<float>()));
    result.charuco_ids.push_back(ids[i].get<int>());
  }
  for (const auto& point : image_points) {
    result.image_points.emplace_back(point.at(0).get<float>(),
                                     point.at(1).get<float>());
  }
  for (const auto& point : object_points) {
    result.object_points.emplace_back(point.at(0).get<float>(),
                                      point.at(1).get<float>(),
                                      point.at(2).get<float>());
  }
  return result;
}

}  // namespace charuco_calibration
