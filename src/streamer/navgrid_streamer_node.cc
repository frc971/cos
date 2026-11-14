#include "streamer/navgrid_streamer_node.h"

#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

auto NavgridStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    cv::Mat new_image = base_image_.clone();
    for (const auto& dependency : dependencies_) {
      if (context->Exists(dependency.GetChannel())) {
        auto* navgrid_message =
            context->GetMessage<NavgridMessage>(dependency.GetChannel());
        DrawNavgrid(new_image, navgrid_message);
      }
    }
    if (!cv::imencode(".jpg", new_image, jpeg_buffer_)) {
      LOG(WARNING) << "Failed to encode";
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }

    const std::string payload(jpeg_buffer_.begin(), jpeg_buffer_.end());
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

void NavgridStreamerNode::DrawNavgrid(cv::Mat& image,
                                      NavgridMessage* navgrid_message) {
  if (!std::isfinite(navgrid_message->field_size_x) ||
      !std::isfinite(navgrid_message->field_size_y) ||
      !std::isfinite(navgrid_message->node_size_meters) ||
      navgrid_message->field_size_x <= 0 ||
      navgrid_message->field_size_y <= 0 ||
      navgrid_message->node_size_meters <= 0) {
    LOG(WARNING) << "Invalid navgrid dimensions";
    return;
  }
  const double cell_width = navgrid_message->node_size_meters * image.cols /
                            navgrid_message->field_size_x;
  const double cell_height = navgrid_message->node_size_meters * image.rows /
                             navgrid_message->field_size_y;
  for (size_t y = 0; y < navgrid_message->navgrid.size(); ++y) {
    const auto& row = navgrid_message->navgrid[y];
    for (size_t x = 0; x < row.size(); ++x) {
      // Navgrid coordinates start at the bottom left of the field.
      const cv::Point top_left(cvRound(x * cell_width),
                               image.rows - cvRound((y + 1) * cell_height));
      const cv::Point bottom_right(cvRound((x + 1) * cell_width) - 1,
                                   image.rows - cvRound(y * cell_height) - 1);
      if (row[x]) {
        cv::rectangle(image, top_left, bottom_right,
                      cv::Scalar(navgrid_message->b, navgrid_message->g,
                                 navgrid_message->r),
                      cv::FILLED);
      }
      cv::rectangle(image, top_left, bottom_right, cv::Scalar(128, 128, 128),
                    1);
    }
  }
}
}  // namespace streamer
