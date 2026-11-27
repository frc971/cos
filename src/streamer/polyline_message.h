#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core/types.hpp>

#include "control_loop/message.h"

namespace streamer {

// A polyline overlay with points and field dimensions in meters.
class PolylineMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(PolylineMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this) +
           (points.size() + markers.size()) * sizeof(cv::Point2d);
  }
  double field_size_x = 0;
  double field_size_y = 0;
  std::uint8_t r = 0;
  std::uint8_t g = 128;
  std::uint8_t b = 255;
  // Render a two-point line as a velocity/direction arrow.
  bool arrow_head = false;
  std::vector<cv::Point2d> points;
  // Highlighted field positions, drawn as magenta circles over the line.
  std::vector<cv::Point2d> markers;
};

}  // namespace streamer
