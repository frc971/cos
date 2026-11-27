#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <opencv4/opencv2/core/mat.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>

#include "control_loop/node.h"
#include "streamer/navgrid_message.h"
#include "streamer/polyline_message.h"

namespace pathing {
class SplinePathMessage;
}  // namespace pathing

namespace localization {
class PathingInputMessage;
}  // namespace localization

namespace streamer {

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

  // Configure live pathing once before registration. Wait for all input
  // channels and render at most once per context and 10 times per second.
  void AddPathingInputs(std::string_view navgrid_channel,
                        std::string_view spline_channel,
                        std::string_view input_channel,
                        std::string_view pose_channel,
                        std::string_view command_channel);

 private:
  struct PathingInputs {
    std::string navgrid;
    std::string spline;
    std::string input;
    std::string pose;
    std::string command;
  };

  void DrawGridRoute(cv::Mat& image, const NavgridMessage& grid,
                     const pathing::SplinePathMessage& spline,
                     const localization::PathingInputMessage* input);
  void DrawPathingVelocity(cv::Mat& image, const NavgridMessage& grid,
                           const control_loop::Context& context);
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
  std::optional<PathingInputs> pathing_inputs_;
  std::mutex frame_mutex_;
  std::optional<std::uint64_t> last_context_id_;
  std::chrono::steady_clock::time_point next_frame_{};
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
