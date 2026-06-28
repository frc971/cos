#pragma once
#include <array>
#include <opencv2/core/types.hpp>
#include <ostream>
#include <vector>

namespace apriltag {

using tag_detection_t = struct TagDetection {
  int tag_id;
  std::array<cv::Point2d, 4> corners;
  double confidence;

  friend auto operator<<(std::ostream& os,
                         const TagDetection& t) -> std::ostream& {
    os << "ID: " << t.tag_id << "\nCorners:\n";
    for (const cv::Point2d& corner : t.corners) {
      os << "(" << corner.x << ", " << corner.y << ")\n";
    }
    return os;
  }
};

}  // namespace apriltag
