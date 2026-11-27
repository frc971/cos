#include "streamer/navgrid_message.h"

#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace streamer {

NavgridMessage::NavgridMessage(double field_size_x, double field_size_y,
                               double node_size_meters,
                               std::vector<std::vector<bool>> grid)
    : field_size_x(field_size_x),
      field_size_y(field_size_y),
      node_size_meters(node_size_meters),
      navgrid(std::move(grid)) {}

auto NavgridMessage::Load(const std::filesystem::path& path) -> NavgridMessage {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open navgrid file: " + path.string());
  }
  const auto json = nlohmann::json::parse(file);
  const auto& field_size = json.at("field_size");
  return NavgridMessage(field_size.at("x").get<double>(),
                        field_size.at("y").get<double>(),
                        json.at("nodeSizeMeters").get<double>(),
                        json.at("grid").get<std::vector<std::vector<bool>>>());
}

}  // namespace streamer
