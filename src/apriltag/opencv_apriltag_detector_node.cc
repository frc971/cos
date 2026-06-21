#include "apriltag/opencv_apriltag_detector_node.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "NvBuffer.h"
#include "absl/log/log.h"
#include "utils/constants_from_json.h"

namespace apriltag {

auto MakeDetector() -> cv::aruco::ArucoDetector {
  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

  cv::aruco::DetectorParameters params;
  params.adaptiveThreshWinSizeMin = 3;
  params.adaptiveThreshWinSizeMax = 53;
  params.adaptiveThreshWinSizeStep = 10;
  params.minMarkerPerimeterRate = 0.02;
  params.maxMarkerPerimeterRate = 4.0;
  params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  params.cornerRefinementWinSize = 5;
  params.cornerRefinementMaxIterations = 30;
  params.cornerRefinementMinAccuracy = 0.1;

  return {dictionary, params};
}

OpenCVApriltagDetectorNode::OpenCVApriltagDetectorNode(
    const nlohmann::json& intrinsics)
    : camera_matrix_(utils::CameraMatrixFromJson<cv::Mat>(intrinsics)),
      distortion_coefficients_(
          utils::DistortionCoefficientsFromJson<cv::Mat>(intrinsics)),
      detector_(MakeDetector()) {}

void OpenCVApriltagDetectorNode::RegisterCallback(
    const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>&
        callback) {
  callbacks_.push_back(callback);
}

void OpenCVApriltagDetectorNode::Detect(
    const camera::DecodedJpegNvBuffer& frame, double timestamp) {
  cv::Mat gray = NvBufferToGray(frame);

  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> rejected;
  detector_.detectMarkers(gray, corners, ids, rejected);

  auto detections = std::make_shared<std::vector<tag_detection_t>>();
  if (!ids.empty()) {
    detections->reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      std::array<cv::Point2d, 4> corners_array;
      for (int j = 0; j < 4; ++j) {
        corners_array[j] = cv::Point2d(corners[i][j].x, corners[i][j].y);
      }
      std::swap(corners_array[0], corners_array[1]);
      std::swap(corners_array[2], corners_array[3]);

      detections->push_back(tag_detection_t{
          .tag_id = ids[i],
          .corners = corners_array,
          .timestamp = timestamp,
          .confidence = 1.0,
      });
    }
  }

  for (const auto& cb : callbacks_) {
    cb(detections);
  }
}

}  // namespace apriltag
