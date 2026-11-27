#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include <opencv2/core/types.hpp>

#include "control_loop/message.h"

namespace streamer {

// A navigation grid in field meters. True cells are occupied in an obstacle
// grid and highlighted in an overlay.
class NavgridMessage final : public control_loop::IMessage {
 public:
  NavgridMessage(double field_size_x, double field_size_y,
                 double node_size_meters, std::vector<std::vector<bool>> grid);

  // Reads the PathPlanner navgrid JSON format.
  static auto Load(const std::filesystem::path& path) -> NavgridMessage;

  auto GetType() -> const std::type_info& override {
    return typeid(NavgridMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }

  double field_size_x;
  double field_size_y;
  double node_size_meters;
  std::uint8_t r = 255;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
  // Indexed by row, then column, starting at the lower-left field corner.
  std::vector<std::vector<bool>> navgrid;
  std::optional<cv::Point> robot_cell;
};

}  // namespace streamer
