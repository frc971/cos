#include "streamer/navgrid_streamer_node.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <turbojpeg.h>

#include "absl/log/log.h"

namespace streamer {
NavgridStreamerNode::NavgridStreamerNode(
    std::string_view path, int port,
    const std::filesystem::path& field_image_path)
    : path_(path),
      port_(port),
      base_image_(cv::imread(field_image_path.string())) {
  if (base_image_.empty()) {
    throw std::runtime_error("Failed to load field image: " +
                             field_image_path.string());
  }
  streamer_.start(port);
  LOG(INFO) << "MJPEG stream available at :" << port << path_;
}

NavgridStreamerNode::NavgridStreamerNode(
    std::string_view path, int port, NavgridMessage navgrid,
    const std::filesystem::path& field_image_path)
    : NavgridStreamerNode(path, port, field_image_path) {
  navgrid_.emplace(std::move(navgrid));
}

NavgridStreamerNode::NavgridStreamerNode(
    std::string_view path, int port, const std::filesystem::path& navgrid_path,
    const std::filesystem::path& field_image_path)
    : NavgridStreamerNode(path, port, NavgridMessage::Load(navgrid_path),
                          field_image_path) {}

auto NavgridStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    cv::Mat new_image = base_image_.clone();
    if (navgrid_)
      DrawNavgrid(new_image, *navgrid_);
    for (const auto& dependency : dependencies_) {
      if (context->Exists(dependency.GetChannel())) {
        if (auto* navgrid =
                context->GetMessage<NavgridMessage>(dependency.GetChannel())) {
          DrawNavgrid(new_image, *navgrid);
        } else if (auto* polyline = context->GetMessage<PolylineMessage>(
                       dependency.GetChannel())) {
          DrawPolyline(new_image, *polyline);
        }
      }
    }
    // OpenCV's jpeg_* calls can bind to NVIDIA's incompatible libnvmm_jpeg
    // ABI when camera FD decoding is linked into the same executable.
    // TurboJPEG exposes tj* entry points and keeps its JPEG implementation
    // private, so hardware decoding and software encoding can coexist.
    const std::unique_ptr<void, decltype(&tjDestroy)> encoder(tjInitCompress(),
                                                           tjDestroy);
    unsigned char* data = nullptr;
    unsigned long size = 0;
    const int result = encoder
        ? tjCompress2(encoder.get(), new_image.data, new_image.cols,
                      static_cast<int>(new_image.step), new_image.rows,
                      TJPF_BGR, &data, &size, TJSAMP_420, 95, 0)
        : -1;
    const std::unique_ptr<unsigned char, decltype(&tjFree)> encoded(data, tjFree);
    if (result != 0) {
      LOG(WARNING) << "Failed to encode";
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }

    const std::string payload(reinterpret_cast<const char*>(encoded.get()), size);
    streamer_.publish(path_, payload);

    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto NavgridStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto NavgridStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void NavgridStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

void NavgridStreamerNode::AddNavgrid(std::string_view input_path) {
  dependencies_.emplace_back(input_path, typeid(NavgridMessage));
}

void NavgridStreamerNode::AddPolyline(std::string_view input_path) {
  dependencies_.emplace_back(input_path, typeid(PolylineMessage));
}

void NavgridStreamerNode::DrawPolyline(cv::Mat& image,
                                       const PolylineMessage& polyline) {
  if (!std::isfinite(polyline.field_size_x) || polyline.field_size_x <= 0 ||
      !std::isfinite(polyline.field_size_y) || polyline.field_size_y <= 0) {
    return;
  }
  std::vector<cv::Point> pixels;
  pixels.reserve(polyline.points.size());
  for (const auto& point : polyline.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 ||
        point.y < 0 || point.x >= polyline.field_size_x ||
        point.y >= polyline.field_size_y) {
      return;
    }
    pixels.emplace_back(
        cvRound(point.x / polyline.field_size_x * (image.cols - 1)),
        image.rows - 1 -
            cvRound(point.y / polyline.field_size_y * (image.rows - 1)));
  }
  const cv::Scalar color(polyline.b, polyline.g, polyline.r);
  if (polyline.arrow_head && pixels.size() == 2) {
    cv::arrowedLine(image, pixels.front(), pixels.back(), cv::Scalar(0, 0, 0),
                    5, cv::LINE_AA, 0, 0.25);
    cv::arrowedLine(image, pixels.front(), pixels.back(), color, 3, cv::LINE_AA,
                    0, 0.25);
    cv::circle(image, pixels.front(), 4, cv::Scalar(0, 0, 0), cv::FILLED,
               cv::LINE_AA);
    cv::circle(image, pixels.front(), 2, color, cv::FILLED, cv::LINE_AA);
  } else if (pixels.size() == 1) {
    cv::circle(image, pixels.front(), 3, color, cv::FILLED, cv::LINE_AA);
  } else if (!pixels.empty()) {
    cv::polylines(image, pixels, false, color, 2, cv::LINE_AA);
  }
  for (const auto& point : polyline.markers) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 ||
        point.y < 0 || point.x >= polyline.field_size_x ||
        point.y >= polyline.field_size_y) {
      continue;
    }
    const cv::Point pixel(
        cvRound(point.x / polyline.field_size_x * (image.cols - 1)),
        image.rows - 1 -
            cvRound(point.y / polyline.field_size_y * (image.rows - 1)));
    cv::circle(image, pixel, 7, cv::Scalar(255, 255, 255), cv::FILLED,
               cv::LINE_AA);
    cv::circle(image, pixel, 5, cv::Scalar(255, 0, 255), cv::FILLED,
               cv::LINE_AA);
    cv::circle(image, pixel, 7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

void NavgridStreamerNode::DrawNavgrid(cv::Mat& image,
                                      const NavgridMessage& navgrid_message) {
  if (!std::isfinite(navgrid_message.field_size_x) ||
      !std::isfinite(navgrid_message.field_size_y) ||
      !std::isfinite(navgrid_message.node_size_meters) ||
      navgrid_message.field_size_x <= 0 ||
      navgrid_message.field_size_y <= 0 ||
      navgrid_message.node_size_meters <= 0) {
    LOG(WARNING) << "Invalid navgrid dimensions";
    return;
  }
  const double cell_width = navgrid_message.node_size_meters * image.cols /
                            navgrid_message.field_size_x;
  const double cell_height = navgrid_message.node_size_meters * image.rows /
                             navgrid_message.field_size_y;
  cv::Rect robot_cell_bounds;
  for (size_t y = 0; y < navgrid_message.navgrid.size(); ++y) {
    const auto& row = navgrid_message.navgrid[y];
    for (size_t x = 0; x < row.size(); ++x) {
      // Navgrid coordinates start at the bottom left of the field.
      const cv::Point top_left(cvRound(x * cell_width),
                               image.rows - cvRound((y + 1) * cell_height));
      const cv::Point bottom_right(cvRound((x + 1) * cell_width) - 1,
                                   image.rows - cvRound(y * cell_height) - 1);
      if (row[x]) {
        cv::rectangle(image, top_left, bottom_right,
                      cv::Scalar(navgrid_message.b, navgrid_message.g,
                                 navgrid_message.r),
                      cv::FILLED);
      }
      cv::rectangle(image, top_left, bottom_right, cv::Scalar(128, 128, 128),
                    1);
      if (navgrid_message.robot_cell ==
          cv::Point(static_cast<int>(x), static_cast<int>(y))) {
        robot_cell_bounds =
            cv::Rect(top_left, bottom_right + cv::Point(1, 1));
      }
    }
  }
  if (!robot_cell_bounds.empty())
    cv::rectangle(image, robot_cell_bounds, cv::Scalar(255, 0, 255), 2);
}
}  // namespace streamer
