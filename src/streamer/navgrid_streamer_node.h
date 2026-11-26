#pragma once

#include <filesystem>
#include <fstream>
#include <opencv4/opencv2/core/mat.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>
#include <nlohmann/json.hpp>

#include "control_loop/node.h"

namespace streamer {

// A polyline overlay with points and field dimensions in meters.
class PolylineMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(PolylineMessage);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) +
           (points.size() + markers.size()) * sizeof(cv::Point2d);
  }
  double field_size_x = 0;
  double field_size_y = 0;
  uint8_t r = 0;
  uint8_t g = 128;
  uint8_t b = 255;
  // Render a two-point line as a velocity/direction arrow.
  bool arrow_head = false;
  std::vector<cv::Point2d> points;
  // Highlighted field positions, drawn as magenta circles over the line.
  std::vector<cv::Point2d> markers;
};

// A PathPlanner navigation grid and its display color. A true cell is drawn
// as part of the overlay. In an obstacle grid, true means occupied.
class NavgridMessage final : public control_loop::IMessage {
 public:
  explicit NavgridMessage(const std::filesystem::path& path)
      : r(255), g(0), b(0) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open navgrid file: " + path.string());
    }
    const auto json = nlohmann::json::parse(file);
    field_size_x = json.at("field_size").at("x").get<double>();
    field_size_y = json.at("field_size").at("y").get<double>();
    node_size_meters = json.at("nodeSizeMeters").get<double>();
    navgrid = json.at("grid").get<std::vector<std::vector<bool>>>();
  }

  auto GetType() -> const std::type_info& override {
    return typeid(NavgridMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }

  double field_size_x;
  double field_size_y;
  double node_size_meters;
  uint8_t r;
  uint8_t g;
  uint8_t b;
  // Indexed by row, then column, starting at the lower-left field corner.
  std::vector<std::vector<bool>> navgrid;
  // Grid indices, with the origin at the bottom left.
  std::optional<cv::Point> robot_cell;
};

// Draws grid and polyline overlays on a field image and serves the result as
// an MJPEG stream. The constructor takes the HTTP path and listen port.
class NavgridStreamerNode final : public control_loop::INode {
 public:
  NavgridStreamerNode(std::string_view path, int port,
                      const std::filesystem::path& field_image_path =
                          "/root/constants/assets/field.png");
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  // Add overlay input channels before registering with the control loop.
  void AddNavgrid(std::string_view input_path);
  void AddPolyline(std::string_view input_path);

 private:
  void DrawNavgrid(cv::Mat& image, NavgridMessage* navgrid_message);
  void DrawPolyline(cv::Mat& image, const PolylineMessage& polyline);

 private:
  std::string path_;
  int port_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  cv::Mat base_image_;
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
