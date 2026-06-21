#pragma once
#include <array>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <ostream>
#include <vector>
#include "camera/nvjpeg_decode_node.h"

namespace apriltag {

using tag_detection_t = struct TagDetection {
  int tag_id;
  std::array<cv::Point2d, 4> corners;
  double timestamp;
  double confidence;

  friend auto operator<<(std::ostream& os,
                         const TagDetection& t) -> std::ostream& {
    os << "ID: " << t.tag_id << "\nCorners:\n";
    for (const cv::Point2d& corner : t.corners) {
      os << "(" << corner.x << ", " << corner.y << ")\n";
    }
    os << "Timestamp: " << t.timestamp << "\nConfidence: " << t.confidence
       << std::endl;
    return os;
  }
};

auto NvBufferToGray(const camera::DecodedJpegNvBuffer& nvbuf) -> cv::Mat {
  const auto& y_plane = nvbuf.buffer->planes[0];
  return cv::Mat(static_cast<int>(y_plane.fmt.height),
                 static_cast<int>(y_plane.fmt.width), CV_8UC1, y_plane.data,
                 y_plane.fmt.stride);
}

class IApriltagDetectorNode {
 public:
  virtual void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>&
          callback) = 0;
  virtual void Detect(const camera::DecodedJpegNvBuffer& frame,
                      double timestamp) = 0;
  virtual ~IApriltagDetectorNode() = default;
};

}  // namespace apriltag
