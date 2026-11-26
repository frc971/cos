#pragma once

#include <chrono>
#include <filesystem>
#include <mutex>
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

  // Field coordinates in meters. The grid must outlive rendering of the context.
  struct PathingFrame {
    const NavgridMessage& grid;
    std::vector<cv::Point2d> route;
    std::optional<cv::Point2d> robot_position;
    std::optional<cv::Point2d> velocity_origin;
    cv::Point2d velocity_mps;
  };
  using PathingFrameSource =
      std::function<std::optional<PathingFrame>(const control_loop::Context&)>;
  // Set the live grid and route source before registering the streamer.
  // Live pathing frames require all input channels and are capped at 10 Hz.
  void SetPathingFrameSource(
      std::vector<control_loop::MessageDescriptor> dependencies,
      PathingFrameSource source);

 private:
  void DrawPathingGrid(cv::Mat& image, const PathingFrame& frame);
  void DrawPathingVelocity(cv::Mat& image, const PathingFrame& frame);
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
  PathingFrameSource pathing_frame_source_;
  std::mutex frame_mutex_;
  std::optional<std::uint64_t> last_context_id_;
  std::chrono::steady_clock::time_point next_frame_{};
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
