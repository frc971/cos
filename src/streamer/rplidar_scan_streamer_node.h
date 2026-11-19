#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>
#include <opencv2/core/mat.hpp>

#include "control_loop/node.h"
#include "pathing/rplidar_node.h"

namespace streamer {

// Separates scan processing and rendering from the HTTP server.
class RPLidarScanRenderer {
 public:
  // max_distance_m must be finite and positive.
  explicit RPLidarScanRenderer(double max_distance_m = 20);
  // nullptr renders a "No scan available" frame.
  [[nodiscard]] auto Render(const pathing::RPLidarMessage* scan) const -> cv::Mat;

 private:
  double max_distance_m_;
};

class RPLidarScanStreamerNode final : public control_loop::INode {
 public:
  RPLidarScanStreamerNode(std::string_view input_path,
                         std::string path = "/lidar", int port = 8080,
                         double max_distance_m = 20);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  void Publish(const pathing::RPLidarMessage* scan);

  std::string input_path_;
  std::string path_;
  RPLidarScanRenderer renderer_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  nadjieb::MJPEGStreamer streamer_;
};

}  // namespace streamer
