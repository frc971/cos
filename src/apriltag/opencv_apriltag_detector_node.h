#pragma once
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include "apriltag/apriltag_detector.h"

namespace apriltag {

class OpenCVApriltagDetectorNode : public IApriltagDetectorNode {
 public:
  explicit OpenCVApriltagDetectorNode(const nlohmann::json& intrinsics);

  void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>,
                               control_loops::MetaDataList metadata,
                               std::shared_ptr<control_loops::Context>)>&
          callback) override;
  void Detect(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx) override;

 private:
  const cv::Mat camera_matrix_;
  const cv::Mat distortion_coefficients_;
  cv::aruco::ArucoDetector detector_;
  std::vector<std::function<void(std::shared_ptr<std::vector<tag_detection_t>>,
                                 control_loops::MetaDataList metadata,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;
};

}  // namespace apriltag
