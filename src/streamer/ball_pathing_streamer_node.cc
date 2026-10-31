#include "streamer/ball_pathing_streamer_node.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "camera/jpeg_buffer.h"

namespace streamer {
namespace {

// Clip in floating point before converting potentially distant projections to
// image coordinates. Near-horizon lens distortion can produce huge pixels.
auto ClipLine(cv::Point2d first, cv::Point2d second, cv::Size size)
    -> std::optional<std::pair<cv::Point, cv::Point>> {
  const auto delta = second - first;
  double low = 0.0;
  double high = 1.0;
  const double p[]{-delta.x, delta.x, -delta.y, delta.y};
  const double q[]{first.x, size.width - 1.0 - first.x,
                   first.y, size.height - 1.0 - first.y};
  for (int i = 0; i < 4; ++i) {
    if (p[i] == 0.0) {
      if (q[i] < 0.0) return std::nullopt;
    } else if (p[i] < 0.0) {
      low = std::max(low, q[i] / p[i]);
    } else {
      high = std::min(high, q[i] / p[i]);
    }
  }
  if (low > high) return std::nullopt;
  const auto pixel = [size](cv::Point2d point) -> cv::Point {
    return {cvRound(std::clamp(point.x, 0.0, size.width - 1.0)),
            cvRound(std::clamp(point.y, 0.0, size.height - 1.0))};
  };
  return std::pair{pixel(first + low * delta), pixel(first + high * delta)};
}

void Label(cv::Mat& image, const std::string& text, int row,
           cv::Scalar color = {255, 255, 255}) {
  const cv::Point origin(12, 24 + row * 25);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
               {0, 0, 0}, 4, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
               color, 1, cv::LINE_AA);
}

}  // namespace

BallPathingOverlay::BallPathingOverlay(const std::string& camera_config,
                                       double radius_m, bool rotate_180)
    : intrinsics_(camera_config),
      extrinsics_(camera_config),
      robot_to_camera_(extrinsics_.ToCameraToRobot<cv::Matx44d>()),
      radius_m_(radius_m),
      rotate_180_(rotate_180) {
  CHECK(std::isfinite(radius_m_) && radius_m_ > 0.0);
}

auto BallPathingOverlay::ProjectFloorPoint(cv::Point2d point) const
    -> std::optional<cv::Point2d> {
  const auto local = robot_to_camera_ * cv::Vec4d{point.x, point.y, 0.0, 1.0};
  if (local[0] <= 0.01) return std::nullopt;
  // Inverse of BallPathingNode's optical-to-WPILib conversion:
  // camera forward/left/up -> OpenCV right/down/forward.
  const double x = -local[1] / local[0];
  const double y = -local[2] / local[0];
  const double r2 = x * x + y * y;
  const double radial = 1 + intrinsics_.k1 * r2 +
                        intrinsics_.k2 * r2 * r2 + intrinsics_.k3 * r2 * r2 * r2;
  const double xd = x * radial + 2 * intrinsics_.p1 * x * y +
                    intrinsics_.p2 * (r2 + 2 * x * x);
  const double yd = y * radial + intrinsics_.p1 * (r2 + 2 * y * y) +
                    2 * intrinsics_.p2 * x * y;
  const cv::Point2d pixel{intrinsics_.fx * xd + intrinsics_.cx,
                           intrinsics_.fy * yd + intrinsics_.cy};
  if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) return std::nullopt;
  return pixel;
}

auto BallPathingOverlay::Render(
    const cv::Mat& image, const gamepiece::BallVelocityGoalMessage* goal) const
    -> cv::Mat {
  cv::Mat output;
  if (rotate_180_) cv::rotate(image, output, cv::ROTATE_180);
  else output = image.clone();
  const auto project = [&](cv::Point2d point) -> std::optional<cv::Point2d> {
    auto pixel = ProjectFloorPoint(point);
    if (pixel && rotate_180_) {
      pixel = cv::Point2d(output.cols - 1 - pixel->x,
                           output.rows - 1 - pixel->y);
    }
    return pixel;
  };
  const bool valid = goal != nullptr && goal->valid;
  const cv::Scalar color = valid && goal->held ? cv::Scalar(0, 200, 255)
                                                : cv::Scalar(70, 255, 40);
  std::ostringstream mount;
  mount << std::fixed << std::setprecision(3)
        << "camera mount xyz=(" << extrinsics_.translation_x << ", "
        << extrinsics_.translation_y << ", " << extrinsics_.translation_z << ") m"
        << std::setprecision(1) << "; rpy=("
        << extrinsics_.rotation_x * 180 / std::numbers::pi << ", "
        << extrinsics_.rotation_y * 180 / std::numbers::pi << ", "
        << extrinsics_.rotation_z * 180 / std::numbers::pi << ") deg";
  Label(output, mount.str(), 0);
  if (!valid) {
    Label(output, "NO TARGET", 1, {0, 100, 255});
    return output;
  }
  const cv::Point2d target(goal->target_x_m, goal->target_y_m);
  std::optional<cv::Point2d> previous;
  for (int i = 0; i <= 720; ++i) {
    const double angle = 2 * std::numbers::pi * i / 720;
    const auto current = project(target + radius_m_ *
        cv::Point2d(std::cos(angle), std::sin(angle)));
    if (previous && current) {
      const auto segment = ClipLine(*previous, *current, output.size());
      if (segment) {
        cv::line(output, segment->first, segment->second, color, 5, cv::LINE_AA);
        cv::line(output, segment->first, segment->second, {255, 255, 255},
                   1, cv::LINE_AA);
      }
    }
    previous = current;
  }
  const auto target_pixel = project(target);
  std::string visibility;
  if (target_pixel && target_pixel->x >= 0 && target_pixel->x < output.cols &&
      target_pixel->y >= 0 && target_pixel->y < output.rows) {
    cv::circle(output, *target_pixel, 10, {210, 40, 255}, -1, cv::LINE_AA);
    cv::circle(output, *target_pixel, 10, {255, 255, 255}, 2, cv::LINE_AA);
  } else {
    visibility = " (off-screen)";
    if (target_pixel) {
      const cv::Point2d center(output.cols / 2.0, output.rows / 2.0);
      const auto segment = ClipLine(center, *target_pixel, output.size());
      if (segment) {
        const cv::Point2d delta = cv::Point2d(segment->second) - center;
        const double length = cv::norm(delta);
        if (length > 0) {
          const cv::Point start = cv::Point2d(segment->second) - delta * (30 / length);
          cv::arrowedLine(output, start, segment->second, {210, 40, 255},
                            3, cv::LINE_AA, 0, 0.4);
        }
      }
    }
  }
  std::ostringstream description;
  description << (goal->held ? "HELD" : "TRACKING") << std::fixed
              << std::setprecision(2) << " target=(" << target.x << ", "
              << target.y << ") m; floor radius=" << radius_m_ << " m" << visibility;
  Label(output, description.str(), 1, color);
  std::ostringstream velocity;
  velocity << std::fixed << std::setprecision(2) << "velocity=("
           << goal->vx_mps << ", " << goal->vy_mps << ") m/s; balls="
           << goal->estimated_ball_count;
  Label(output, velocity.str(), 2, color);
  return output;
}

BallPathingStreamerNode::BallPathingStreamerNode(
    std::string_view jpeg_path, std::string_view goal_path,
    const std::string& camera_config, double radius_m, std::string stream_path,
    int port, bool rotate_180, double maximum_fps)
    : overlay_(camera_config, radius_m, rotate_180),
      jpeg_path_(jpeg_path),
      goal_path_(goal_path),
      stream_path_(std::move(stream_path)),
      dependencies_({{jpeg_path_, typeid(camera::JpegBuffer)},
                      {goal_path_, typeid(gamepiece::BallVelocityGoalMessage)}}) {
  CHECK(std::isfinite(maximum_fps) && maximum_fps > 0.0);
  frame_interval_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / maximum_fps));
  streamer_.start(port);
  LOG(INFO) << "Ball pathing MJPEG stream available at :" << port << stream_path_;
}

BallPathingStreamerNode::~BallPathingStreamerNode() { worker_.Shutdown(); }

void BallPathingStreamerNode::Notify(const control_loop::Context& context) const {
  for (const auto& callback : callbacks_) callback(context);
}

auto BallPathingStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    const auto jpeg = context->GetSharedMessage<camera::JpegBuffer>(jpeg_path_);
    if (!jpeg || !jpeg->ptr || jpeg->size == 0 ||
        jpeg->size > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        busy_.exchange(true)) {
      Notify(context);
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_frame_ < frame_interval_) {
      busy_.store(false);
      Notify(context);
      return;
    }
    last_frame_ = now;
    const auto goal =
        context->GetSharedMessage<gamepiece::BallVelocityGoalMessage>(goal_path_);
    worker_.Submit([this, jpeg, goal] {
      try {
        const cv::Mat encoded(1, static_cast<int>(jpeg->size), CV_8UC1, jpeg->ptr);
        const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
        if (!decoded.empty()) {
          const cv::Mat annotated = overlay_.Render(decoded, goal.get());
          std::vector<unsigned char> output;
          if (cv::imencode(".jpg", annotated, output,
                           {cv::IMWRITE_JPEG_QUALITY, 85})) {
            streamer_.publish(stream_path_, std::string(
                reinterpret_cast<const char*>(output.data()), output.size()));
          }
        }
      } catch (const cv::Exception& error) {
        LOG(WARNING) << "Unable to render ball pathing stream: " << error.what();
      }
      busy_.store(false);
    }, context->id);
    // Keep the frame and goal alive, not the context: JPEG rendering must not
    // occupy one of the control loop's limited in-flight context slots.
    Notify(context);
  };
}

void BallPathingStreamerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto BallPathingStreamerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto BallPathingStreamerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace streamer
