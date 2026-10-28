#include "streamer/cluster_debug_streamer_node.h"

#include <cmath>
#include <numbers>
#include <opencv2/imgproc.hpp>
#include "localization/solver_common.h"
#include "streamer/cluster_geometry.h"

namespace streamer {
namespace {
constexpr double kScale = 80, kBallRadiusM = 0.075;  // Matches HSVClusterTrackerNode::kFuelRadiusM.
auto Point(double x, double y) -> cv::Point {
  return {cvRound(x * kScale), cvRound((localization::kApriltagLayout.GetFieldWidth().value() - y) * kScale)};
}
auto Canvas(const localization::PositionEstimateMessage* pose) -> cv::Mat {
  const auto& layout = localization::kApriltagLayout;
  cv::Mat image(cvRound(layout.GetFieldWidth().value() * kScale),
                cvRound(layout.GetFieldLength().value() * kScale), CV_8UC3, cv::Scalar{40, 40, 40});
  for (int x = 0; x <= layout.GetFieldLength().value(); ++x)
    cv::line(image, Point(x, 0), Point(x, layout.GetFieldWidth().value()), {70, 70, 70});
  for (int y = 0; y <= layout.GetFieldWidth().value(); ++y)
    cv::line(image, Point(0, y), Point(layout.GetFieldLength().value(), y), {70, 70, 70});
  for (const auto& tag : layout.GetTags())
    cv::circle(image, Point(tag.pose.X().value(), tag.pose.Y().value()), 3, {255, 255, 255}, cv::FILLED);
  if (pose) {
    const auto p = pose->pose.ToPose2d();
    const double x = p.X().value(), y = p.Y().value(), yaw = p.Rotation().Radians().value();
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(yaw)) {
      cv::circle(image, Point(x, y), 10, {255, 200, 0}, 2);
      cv::arrowedLine(image, Point(x, y), Point(x + .5 * std::cos(yaw), y + .5 * std::sin(yaw)), {255, 200, 0}, 2);
    }
  }
  return image;
}
void Label(cv::Mat& image, const std::string& text) {
  cv::putText(image, text, {16, 25}, cv::FONT_HERSHEY_SIMPLEX, .6, {255, 255, 255}, 1, cv::LINE_AA);
}
}  // namespace

ClusterDebugStreamerNode::ClusterDebugStreamerNode(std::string clusters, std::string pose,
    std::string densest, std::string mask, int port)
    : clusters_(std::move(clusters)), pose_(std::move(pose)),
      densest_(std::move(densest)), mask_(std::move(mask)),
      dependencies_{{clusters_, typeid(control_loop::ValueMessage<gamepiece::hsv_clusters_t>)},
                    {pose_, typeid(localization::PositionEstimateMessage)},
                    {densest_, typeid(control_loop::ValueMessage<std::vector<double>>)},
                    {mask_, typeid(control_loop::ValueMessage<cv::Mat>)}} {
  streamer_.start(port, 2);
  cv::Mat waiting(240, 640, CV_8UC3, cv::Scalar{40, 40, 40});
  Label(waiting, "Waiting for a camera frame");
  const auto initial = encoder_.Encode(waiting);
  for (const auto* path : {"/stream", "/hsv", "/clusters", "/balls"}) streamer_.publish(path, initial);
}

auto ClusterDebugStreamerNode::CreateCallback() -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    {
      std::scoped_lock lock(mutex_);
      for (const auto& dependency : dependencies_)
        if (!context->Exists(dependency.GetChannel())) return;
      const auto* clusters = context->GetMessage<control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(clusters_);
      if (!clusters) return;
      // Several dependencies may trigger us in the same context.
      const std::string done = mask_ + "/streamed";
      if (context->Exists(done)) return;
      context->SetMessage(done, nullptr);
      auto publish = [&](const std::string& path, const cv::Mat& image) {
        if (image.empty() || !streamer_.hasClient(path)) return;
        streamer_.publish(path, encoder_.Encode(image));
      };
      if (const auto* mask = context->GetMessage<control_loop::ValueMessage<cv::Mat>>(mask_))
        publish("/hsv", mask->value);
      const auto* pose = context->GetMessage<localization::PositionEstimateMessage>(pose_);
      if (streamer_.hasClient("/stream")) {
        auto image = Canvas(pose);
        Label(image, !pose ? "Waiting for /Drive/Pose" : !clusters ? "No camera frame" : "Live 2-sigma ellipses | field meters | 1 m grid");
        if (pose && clusters) for (const auto& cluster : clusters->value) {
          const auto c = detail::FieldCluster(cluster, pose->pose.ToPose2d());
          if (!c) continue;
          cv::ellipse(image, Point(c->center_x, c->center_y),
              {cvRound(c->semimajor_length * kScale), cvRound(c->semiminor_length * kScale)},
              -c->field_relative_semimajor_axis_yaw * 180 / std::numbers::pi, 0, 360, {0, 220, 255}, 2, cv::LINE_AA);
          cv::putText(image, cv::format("%zu: %.1f balls/m2", cluster.id, c->predicted_density),
              Point(c->center_x, c->center_y), cv::FONT_HERSHEY_SIMPLEX, .45, {255, 255, 255}, 1, cv::LINE_AA);
        }
        publish("/stream", image);
      }
      if (streamer_.hasClient("/clusters")) {
        auto image = Canvas(pose);
        const auto* output = context->GetMessage<control_loop::ValueMessage<std::vector<double>>>(densest_);
        const auto* v = output ? &output->value : nullptr;
        Label(image, v && v->size() == 10 ? "/COS/DensestCluster | center + major/minor endpoints (meters)" : "/COS/DensestCluster: empty (no valid cluster/pose)");
        if (v && v->size() == 10) {
          auto point = [&](int i) { return Point((*v)[i], (*v)[i + 1]); };
          const auto center = point(0), major = point(2), minor = point(6);
          cv::ellipse(image, center, {cvRound(cv::norm(major - center)), cvRound(cv::norm(minor - center))},
              std::atan2(major.y - center.y, major.x - center.x) * 180 / std::numbers::pi,
              0, 360, {0, 220, 255}, 2, cv::LINE_AA);
          cv::line(image, major, point(4), {0, 220, 255});
          cv::line(image, minor, point(8), {0, 220, 255});
          for (int i = 0; i < 10; i += 2) {
            cv::circle(image, point(i), 4, {0, 255, 0}, cv::FILLED);
            cv::putText(image, cv::format("(%.2f, %.2f)", (*v)[i], (*v)[i + 1]), point(i) + cv::Point{6, -6},
                cv::FONT_HERSHEY_SIMPLEX, .4, {255, 255, 255}, 1, cv::LINE_AA);
          }
        }
        publish("/clusters", image);
      }
      if (pose && streamer_.hasClient("/balls")) {
        auto image = Canvas(pose);
        const auto robot = pose->pose.ToPose2d();
        size_t count = 0;
        for (const auto& cluster : clusters->value)
          for (const auto& ball : cluster.verified_ball_centers) {
            const auto field = frc::Translation2d{units::meter_t{ball.x}, units::meter_t{ball.y}}
                                   .RotateBy(robot.Rotation()) + robot.Translation();
            cv::circle(image, Point(field.X().value(), field.Y().value()),
                       cvRound(kBallRadiusM * kScale), {0, 220, 255}, 2, cv::LINE_AA);
            ++count;
          }
        Label(image, cv::format("%zu ball detections | field meters | 1 m grid", count));
        publish("/balls", image);
      }
    }
    for (const auto& callback : callbacks_) callback(context);
  };
}
}  // namespace streamer
