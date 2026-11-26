#include "streamer/navgrid_streamer_node.h"

#include <cmath>
#include <memory>
#include <ranges>
#include <algorithm>
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
    std::unique_lock lock(frame_mutex_);
    std::optional<PathingFrame> frame;
    if (pathing_frame_source_) {
      if (last_context_id_ && context->id <= *last_context_id_)
        return;
      for (const auto& dependency : dependencies_) {
        if (!context->Exists(dependency.GetChannel()))
          return;
      }
      last_context_id_ = context->id;
      const auto now = std::chrono::steady_clock::now();
      if (now < next_frame_)
        return;
      auto candidate = pathing_frame_source_(context);
      if (!candidate)
        return;
      frame.emplace(std::move(*candidate));
      next_frame_ = now + std::chrono::milliseconds(100);
    }
    cv::Mat new_image = base_image_.clone();
    if (frame)
      DrawPathingGrid(new_image, *frame);
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
    if (frame)
      DrawPathingVelocity(new_image, *frame);
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
    } else {
      const std::string payload(reinterpret_cast<const char*>(encoded.get()),
                                size);
      streamer_.publish(path_, payload);
    }
    lock.unlock();

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

void NavgridStreamerNode::SetPathingFrameSource(
    std::vector<control_loop::MessageDescriptor> dependencies,
    PathingFrameSource source) {
  for (auto& dependency : dependencies)
    dependencies_.push_back(std::move(dependency));
  pathing_frame_source_ = std::move(source);
}

void NavgridStreamerNode::DrawPathingGrid(cv::Mat& image,
                                          const PathingFrame& frame) {
  const auto& grid = frame.grid;
  DrawNavgrid(image, grid);
  if (!std::isfinite(grid.node_size_meters) || grid.node_size_meters <= 0)
    return;

  auto route = grid;
  route.r = route.b = 0;
  route.g = 255;
  for (auto& row : route.navgrid)
    std::ranges::fill(row, false);

  const auto cell_for = [&](cv::Point2d point) -> std::optional<cv::Point> {
    if (!(point.x >= 0 && point.x < grid.field_size_x &&
          point.y >= 0 && point.y < grid.field_size_y))
      return std::nullopt;
    const double col = point.x / grid.node_size_meters;
    const double row = point.y / grid.node_size_meters;
    if (row >= route.navgrid.size() ||
        col >= route.navgrid[static_cast<size_t>(row)].size())
      return std::nullopt;
    return cv::Point(static_cast<int>(col), static_cast<int>(row));
  };
  for (const auto& point : frame.route) {
    if (const auto cell = cell_for(point))
      route.navgrid[cell->y][cell->x] = true;
  }
  if (frame.robot_position)
    route.robot_cell = cell_for(*frame.robot_position);
  DrawNavgrid(image, route);
}

void NavgridStreamerNode::DrawPathingVelocity(cv::Mat& image,
                                              const PathingFrame& frame) {
  if (!frame.velocity_origin)
    return;
  PolylineMessage velocity;
  velocity.field_size_x = frame.grid.field_size_x;
  velocity.field_size_y = frame.grid.field_size_y;
  velocity.g = velocity.b = 255;
  velocity.arrow_head = true;
  // The arrow shows one second of commanded translation.
  velocity.points = {*frame.velocity_origin,
                     *frame.velocity_origin + frame.velocity_mps};
  DrawPolyline(image, velocity);
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
