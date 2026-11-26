#pragma once

#include <filesystem>
#include <opencv4/opencv2/core/mat.hpp>
#include <optional>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>

#include "control_loop/node.h"
#include "streamer/navgrid_message.h"

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

// Draws grid and polyline overlays on a field image and serves the result as
// an MJPEG stream. The constructor takes the HTTP path and listen port.
class NavgridStreamerNode final : public control_loop::INode {
 public:
  NavgridStreamerNode(std::string_view path, int port,
                      const std::filesystem::path& field_image_path =
                          "/root/constants/assets/field.png");
  // Draw a fixed navgrid beneath the overlays supplied by input channels.
  NavgridStreamerNode(std::string_view path, int port, NavgridMessage navgrid,
                      const std::filesystem::path& field_image_path =
                          "/root/constants/assets/field.png");
  // Load the fixed navgrid from PathPlanner JSON.
  NavgridStreamerNode(std::string_view path, int port,
                      const std::filesystem::path& navgrid_path,
                      const std::filesystem::path& field_image_path);
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
  void DrawNavgrid(cv::Mat& image, const NavgridMessage& navgrid_message);
  void DrawPolyline(cv::Mat& image, const PolylineMessage& polyline);

 private:
  std::string path_;
  int port_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  cv::Mat base_image_;
  std::optional<NavgridMessage> navgrid_;
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
