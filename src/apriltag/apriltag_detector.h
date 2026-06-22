#pragma once
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>
#include "apriltag/tag_detection.h"
#include "camera/nvjpeg_decode_node.h"

namespace apriltag {

auto NvBufferToGray(const camera::DecodedJpegNvBuffer& nvbuf) -> cv::Mat {
  const auto& y_plane = nvbuf.buffer->planes[0];
  return {static_cast<int>(y_plane.fmt.height),
          static_cast<int>(y_plane.fmt.width), CV_8UC1, y_plane.data,
          y_plane.fmt.stride};
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
