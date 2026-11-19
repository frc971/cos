#include "streamer/rplidar_scan_streamer_node.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/log.h"

namespace streamer {

RPLidarScanRenderer::RPLidarScanRenderer(double max_distance_m)
    : max_distance_m_(max_distance_m) {
  if (!std::isfinite(max_distance_m) || max_distance_m <= 0) {
    throw std::invalid_argument("Scan view requires a finite positive range");
  }
}

auto RPLidarScanRenderer::Render(const pathing::RPLidarMessage* scan) const
    -> cv::Mat {
  cv::Mat image(540, 960, CV_8UC3, cv::Scalar(24, 24, 24));
  const cv::Scalar text(230, 230, 230);
  const cv::Scalar grid(75, 75, 75);
  const cv::Scalar zero_color(0, 255, 255);
  const cv::Point center(480, 285);
  constexpr int radius = 200;
  const auto label = [&](const std::string& value, int x, int y,
                         double scale = 0.5) {
    cv::putText(image, value, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale,
                text, 1, cv::LINE_AA);
  };
  label("LiDAR scan", 80, 32, 0.8);
  cv::putText(image, "Yellow rim: no distance return", {600, 32},
              cv::FONT_HERSHEY_SIMPLEX, 0.5, zero_color, 1, cv::LINE_AA);
  for (int ring = 1; ring <= 4; ++ring) {
    const int r = radius * ring / 4;
    cv::circle(image, center, r, grid, 1, cv::LINE_AA);
    label(std::format("{:.3g} m", max_distance_m_ * (ring / 4.0)),
          center.x + r + 5, center.y - 8);
  }
  cv::line(image, {center.x - radius, center.y},
            {center.x + radius, center.y}, grid);
  cv::line(image, {center.x, center.y - radius},
            {center.x, center.y + radius}, grid);
  label("0 deg", center.x - 20, center.y - radius - 10);
  label("90 deg", center.x + radius + 15, center.y + 20);
  label("180 deg", center.x - 30, center.y + radius + 20);
  label("270 deg", center.x - radius - 85, center.y + 20);

  std::size_t samples = 0, plotted = 0, zero = 0, out_of_range = 0;
  if (scan != nullptr) {
    samples = std::min(scan->length, scan->nodes.size());
    for (std::size_t i = 0; i < samples; ++i) {
      const auto& node = scan->nodes[i];
      const double distance = node.dist_mm_q2 / 4000.0;
      const bool is_zero = distance == 0;
      if (distance > max_distance_m_) {
        ++out_of_range;
        continue;
      }
      // SDK angles are Q14 quarter-turns. Show 0 degrees up, clockwise.
      const double angle = node.angle_z_q14 * std::numbers::pi / 32768.0;
      const double r = is_zero ? radius : (distance / max_distance_m_) * radius;
      const cv::Point pixel(center.x + cvRound(r * std::sin(angle)),
                            center.y - cvRound(r * std::cos(angle)));
      cv::circle(image, pixel, 2,
                 is_zero ? zero_color : cv::Scalar(220, 170, 60), cv::FILLED,
                 cv::LINE_AA);
      if (is_zero) {
        ++zero;
      } else {
        ++plotted;
      }
    }
  }
  cv::drawMarker(image, center, text, cv::MARKER_CROSS, 10, 1, cv::LINE_AA);
  label(std::format("Samples: {}   Plotted: {}   Zero: {}   Out of range: {}",
                    samples, plotted, zero, out_of_range), 80, 58);
  label("Sensor at center | clockwise angles | current scan", 80, 530);
  if (scan == nullptr) {
    label("No scan available", 370, 320, 0.7);
  } else if (plotted == 0) {
    label("No usable returns", 370, 320, 0.7);
  }
  return image;
}

RPLidarScanStreamerNode::RPLidarScanStreamerNode(
    std::string_view input_path, std::string path, int port,
    double max_distance_m)
    : input_path_(input_path),
      path_(std::move(path)),
      renderer_(max_distance_m),
      dependencies_({{input_path_, typeid(pathing::RPLidarMessage)}}) {
  if (input_path_.empty() || port < 1 || port > 65535 || path_.empty() ||
      path_.front() != '/' || path_ == "/shutdown" ||
      path_.find_first_of(" \t\r\n?#") != std::string::npos) {
    throw std::invalid_argument("Invalid scan input channel, port or URL path");
  }
  streamer_.start(port);
  // Register the URL before the first scan arrives.
  Publish(nullptr);
  LOG(INFO) << "LiDAR scan available at http://<robot-host>:" << port << path_;
}

void RPLidarScanStreamerNode::Publish(const pathing::RPLidarMessage* scan) {
  try {
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(".jpg", renderer_.Render(scan), jpeg)) {
      LOG(WARNING) << "Failed to encode LiDAR scan";
      return;
    }
    streamer_.publish(path_, std::string(jpeg.begin(), jpeg.end()));
  } catch (const cv::Exception& error) {
    LOG(WARNING) << "Failed to render LiDAR scan: " << error.what();
  }
}

auto RPLidarScanStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    Publish(context->GetMessage<pathing::RPLidarMessage>(input_path_));
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto RPLidarScanStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto RPLidarScanStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void RPLidarScanStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace streamer
