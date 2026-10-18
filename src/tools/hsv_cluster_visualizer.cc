#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <frc/apriltag/AprilTagDetector.h>
#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "camera/camera_config.h"
#include "control_loop/thread_pool.h"
#include "localization/multi_tag_solver_node.h"
#include "utils/cv_geometry.h"

namespace {

constexpr std::string_view kDefaultIntrinsicsPath =
    "constants/gamepiece/camera.json";
constexpr std::string_view kDefaultExtrinsicsPath =
    "constants/gamepiece/camera.json";
constexpr std::string_view kDefaultOutputDirectory =
    "/root/debug-output/hsv-cluster-visualizer";
constexpr std::string_view kDefaultFramesDirectory =
    "/cos-logs/downloaded_video_frames";
constexpr std::string_view kDefaultFramePath =
    "/cos-logs/downloaded_video_frames/frame_001500.jpg";

// Futures propagate worker exceptions and bound the number of in-flight frames.
template <typename F>
auto Async(control_loop::ThreadPool& pool, F task) {
  auto work = std::make_shared<std::packaged_task<decltype(task())()>>(
      std::move(task));
  auto result = work->get_future();
  pool.Submit([work] { (*work)(); });
  return result;
}

// Drawing helpers are adapted from BOS's hsv_cluster_visualizer.cc.
constexpr std::mt19937::result_type kClusterColorSeed = 0x4B4D4541;

auto HueColor(const int hue) -> cv::Scalar {
  const cv::Mat hsv_pixel(1, 1, CV_8UC3,
                          cv::Scalar{static_cast<double>(hue), 220.0, 255.0});
  cv::Mat bgr_pixel;
  cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
  const cv::Vec3b color = bgr_pixel.at<cv::Vec3b>(0, 0);
  return {static_cast<double>(color[0]), static_cast<double>(color[1]),
          static_cast<double>(color[2])};
}

auto ClusterColor(std::size_t id) -> cv::Scalar {
  std::mt19937 random_generator(kClusterColorSeed + id);
  return HueColor(std::uniform_int_distribution<int>(0, 179)(random_generator));
}

// Top-down field view in WPILib axes: +X right along the field, +Y up.
constexpr double kFieldPixelsPerMeter = 80.0;
constexpr double kFieldEllipseAlpha = 0.3;

auto FieldPoint(const double x_m, const double y_m) -> cv::Point2f {
  return {static_cast<float>(x_m * kFieldPixelsPerMeter),
          static_cast<float>(
              (localization::kApriltagLayout.GetFieldWidth().value() - y_m) *
              kFieldPixelsPerMeter)};
}

auto MakeFieldCanvas() -> cv::Mat {
  const auto& layout = localization::kApriltagLayout;
  cv::Mat3b canvas(
      static_cast<int>(layout.GetFieldWidth().value() * kFieldPixelsPerMeter),
      static_cast<int>(layout.GetFieldLength().value() * kFieldPixelsPerMeter),
      cv::Vec3b{40, 40, 40});
  for (int x = 0; x <= static_cast<int>(layout.GetFieldLength().value()); ++x) {
    cv::line(canvas, FieldPoint(x, 0),
             FieldPoint(x, layout.GetFieldWidth().value()),
             cv::Scalar{70, 70, 70}, 1);
  }
  for (int y = 0; y <= static_cast<int>(layout.GetFieldWidth().value()); ++y) {
    cv::line(canvas, FieldPoint(0, y),
             FieldPoint(layout.GetFieldLength().value(), y),
             cv::Scalar{70, 70, 70}, 1);
  }
  for (const frc::AprilTag& tag : layout.GetTags()) {
    cv::circle(canvas, FieldPoint(tag.pose.X().value(), tag.pose.Y().value()),
               3, cv::Scalar{255, 255, 255}, cv::FILLED, cv::LINE_AA);
  }
  return canvas;
}

auto DrawRobot(cv::Mat& canvas, const frc::Pose2d& pose,
               const cv::Scalar& color) -> void {
  const frc::Translation2d nose =
      pose.Translation() +
      frc::Translation2d{0.5_m, 0_m}.RotateBy(pose.Rotation());
  cv::circle(canvas, FieldPoint(pose.X().value(), pose.Y().value()), 5, color,
             cv::FILLED, cv::LINE_AA);
  cv::line(canvas, FieldPoint(pose.X().value(), pose.Y().value()),
           FieldPoint(nose.X().value(), nose.Y().value()), color, 2,
           cv::LINE_AA);
}

auto ProjectedPointCounts(const gamepiece::HSVClusterTrackerNode& tracker,
                          const gamepiece::hsv_clusters_t& clusters,
                          const frc::Pose2d& pose, cv::Size size) -> cv::Mat1f {
  cv::Mat1f counts(size, 0.0f);
  for (const auto& cluster : clusters) {
    for (const auto& point : cluster.img_points) {
      if (const auto offset = tracker.RobotRelativeOffset(point)) {
        const auto position = pose.Translation() + offset->RotateBy(pose.Rotation());
        const auto pixel = FieldPoint(position.X().value(), position.Y().value());
        if (pixel.x >= 0 && pixel.x < size.width &&
            pixel.y >= 0 && pixel.y < size.height) {
          counts(static_cast<int>(pixel.y), static_cast<int>(pixel.x)) += 1.0f;
        }
      }
    }
  }
  return counts;
}

auto PointHeatmap(const cv::Mat& base, const cv::Mat1f& counts,
                  const std::string& caption) -> cv::Mat {
  // Smooth over 5 cm, then use log density to keep sparse projections visible.
  cv::Mat1f density, intensity;
  cv::GaussianBlur(counts, density, {}, 0.05 * kFieldPixelsPerMeter,
                   0.05 * kFieldPixelsPerMeter, cv::BORDER_CONSTANT);
  double peak = 0;
  cv::minMaxLoc(density, nullptr, &peak);
  cv::log(density + 1.0f, intensity);
  cv::Mat1b levels;
  intensity.convertTo(levels, CV_8U, peak > 0 ? 255.0 / std::log1p(peak) : 0);
  cv::Mat colors, blended, heatmap = base.clone();
  cv::applyColorMap(levels, colors, cv::COLORMAP_TURBO);
  cv::addWeighted(colors, 0.85, base, 0.15, 0, blended);
  blended.copyTo(heatmap, density > 0.001f);
  cv::putText(heatmap, caption, {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar{255, 255, 255}, 2, cv::LINE_AA);
  cv::putText(heatmap,
              cv::format("%.0f projected pixels on field | 5 cm smoothing",
                         cv::sum(counts)[0]),
              {20, 58}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar{255, 255, 255}, 1, cv::LINE_AA);
  cv::Mat1b ramp(16, 256);
  for (int x = 0; x < ramp.cols; ++x) ramp.col(x).setTo(x);
  cv::Mat legend;
  cv::applyColorMap(ramp, legend, cv::COLORMAP_TURBO);
  legend.copyTo(heatmap(cv::Rect{20, heatmap.rows - 42, ramp.cols, ramp.rows}));
  cv::putText(heatmap,
              cv::format("0 to %.2f pixels/cell (log scale, per-image peak)", peak),
              {20, heatmap.rows - 52}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar{255, 255, 255}, 1, cv::LINE_AA);
  return heatmap;
}

// The cluster's ground-plane 2-sigma ellipse (matching ClusterEllipse's
// padding-free diameter), centered at the projected centroid, in field-canvas
// pixels. Robot-relative geometry is moved onto the field by `robot_pose`.
auto FieldEllipse(const gamepiece::HSVClusterTrackerNode& tracker,
                  const gamepiece::kmeans_cluster_t& cluster,
                  const frc::Pose2d& robot_pose)
    -> std::optional<cv::RotatedRect> {
  const auto center = tracker.RobotRelativeOffset(cluster.centroid);
  std::vector<cv::Point2f> ground_points;
  for (const cv::Point2f& point : cluster.img_points) {
    if (const auto offset = tracker.RobotRelativeOffset(point)) {
      ground_points.emplace_back(static_cast<float>(offset->X().value()),
                                 static_cast<float>(offset->Y().value()));
    }
  }
  if (!center.has_value() || ground_points.size() < 2) {
    return std::nullopt;
  }
  cv::Mat covariance;
  cv::Mat mean;
  cv::calcCovarMatrix(cv::Mat(ground_points).reshape(1), covariance, mean,
                      cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE,
                      CV_32F);
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::eigen(covariance, eigenvalues, eigenvectors);
  const auto diameter = [&](const int i) -> float {
    return static_cast<float>(
        4.0 * kFieldPixelsPerMeter *
        std::sqrt(std::max(eigenvalues.at<float>(i, 0), 0.0f)));
  };
  // Field-axis angle of the major axis; the canvas's +Y is image-up.
  const double field_angle =
      std::atan2(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(0, 0)) +
      robot_pose.Rotation().Radians().value();
  const frc::Translation2d field_center =
      robot_pose.Translation() + center->RotateBy(robot_pose.Rotation());
  return cv::RotatedRect{
      FieldPoint(field_center.X().value(), field_center.Y().value()),
      {diameter(0), diameter(1)},
      static_cast<float>(-field_angle * 180.0 / std::numbers::pi)};
}

auto DrawTranslucentEllipse(cv::Mat& canvas, const cv::RotatedRect& ellipse,
                            const cv::Scalar& color) -> void {
  cv::Mat filled = canvas.clone();
  cv::ellipse(filled, ellipse, color, cv::FILLED, cv::LINE_AA);
  cv::addWeighted(filled, kFieldEllipseAlpha, canvas, 1.0 - kFieldEllipseAlpha,
                  0.0, canvas);
  cv::ellipse(canvas, ellipse, color, 1, cv::LINE_AA);
}

auto PixelEllipse(const gamepiece::Ellipse& ellipse,
                  const cv::Mat& camera_matrix) -> cv::RotatedRect {
  // Transform the algorithm's actual ellipse, including its normalized radius
  // padding. Unequal fx/fy can change both its axis lengths and orientation.
  const cv::Matx33d intrinsics(camera_matrix);
  const double cosine = std::cos(ellipse.rotation);
  const double sine = std::sin(ellipse.rotation);
  const cv::Matx22d axes{intrinsics(0, 0) * ellipse.semi_axes.width * cosine,
                         -intrinsics(0, 0) * ellipse.semi_axes.height * sine,
                         intrinsics(1, 1) * ellipse.semi_axes.width * sine,
                         intrinsics(1, 1) * ellipse.semi_axes.height * cosine};
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::eigen(cv::Mat(axes * axes.t()), eigenvalues, eigenvectors);
  const double angle =
      std::atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0)) *
      180.0 / std::numbers::pi;
  const auto diameter = [](double eigenvalue) -> float {
    return static_cast<float>(2.0 * std::sqrt(std::max(eigenvalue, 0.0)));
  };
  return {{static_cast<float>(ellipse.center.x * intrinsics(0, 0) +
                              intrinsics(0, 2)),
           static_cast<float>(ellipse.center.y * intrinsics(1, 1) +
                              intrinsics(1, 2))},
          {diameter(eigenvalues.at<double>(0, 0)),
           diameter(eigenvalues.at<double>(1, 0))},
          static_cast<float>(angle)};
}

auto NearestClusterDistance(
    const std::vector<gamepiece::kmeans_cluster_t>& clusters,
    const std::size_t index) -> double {
  double nearest = std::numeric_limits<double>::infinity();
  const auto& position = clusters[index].robot_relative_translation;
  for (std::size_t i = 0; position.has_value() && i < clusters.size(); ++i) {
    const auto& other = clusters[i].robot_relative_translation;
    if (i != index && other.has_value()) {
      nearest = std::min(nearest, position->Distance(*other).value());
    }
  }
  return nearest;
}

auto DrawClusterIndex(cv::Mat& image, const std::size_t cluster_index,
                      const cv::Point& centroid,
                      const double nearest_distance,
                      std::vector<cv::Rect>* occupied = nullptr) -> void {
  const std::string label = std::isfinite(nearest_distance)
                                ? cv::format("%zu (%.2f m)", cluster_index,
                                             nearest_distance)
                                : cv::format("%zu (N/A)", cluster_index);
  constexpr double kFontScale = 0.65;
  constexpr int kTextThickness = 2;
  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(
      label, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kTextThickness, &baseline);
  cv::Point text_origin{
      std::clamp(centroid.x - text_size.width / 2, 4,
                 std::max(4, image.cols - text_size.width - 4)),
      std::clamp(centroid.y + text_size.height / 2, text_size.height + 4,
                 image.rows - baseline - 4)};
  if (occupied != nullptr) {
    const auto bounds = [&] {
      return cv::Rect{text_origin.x - 4,
                      text_origin.y - text_size.height - 4,
                      text_size.width + 8, text_size.height + baseline + 8};
    };
    while (text_origin.y > text_size.height + 4 &&
           std::ranges::any_of(*occupied, [&](const cv::Rect& previous) {
             return (previous & bounds()).area() > 0;
           })) {
      text_origin.y = std::max(text_size.height + 4,
                              text_origin.y - text_size.height - baseline - 8);
    }
    occupied->push_back(bounds());
  }

  cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, kFontScale,
              cv::Scalar{0, 0, 0}, kTextThickness + 4, cv::LINE_AA);
  cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, kFontScale,
              cv::Scalar{255, 255, 255}, kTextThickness, cv::LINE_AA);
}

// Convert normalized geometry to the undistorted image's pixel coordinates.
auto PixelCluster(const gamepiece::kmeans_cluster_t& cluster,
                  const cv::Mat& camera_matrix) -> gamepiece::kmeans_cluster_t {
  const cv::Matx33f intrinsics(camera_matrix);
  const cv::Matx22f scale{intrinsics(0, 0), 0, 0, intrinsics(1, 1)};
  const auto pixel_point = [&](const cv::Point2f& point) -> cv::Point2f {
    return {point.x * intrinsics(0, 0) + intrinsics(0, 2),
            point.y * intrinsics(1, 1) + intrinsics(1, 2)};
  };
  gamepiece::kmeans_cluster_t result = cluster;
  result.centroid = pixel_point(cluster.centroid);
  result.covar = scale * cluster.covar * scale.t();
  for (auto& point : result.img_points) {
    point = pixel_point(point);
  }
  return result;
}

auto AnnotateFrame(const cv::Mat& frame,
                   const std::vector<gamepiece::kmeans_cluster_t>& clusters,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& map_x, const cv::Mat& map_y,
                   const double pitch)
    -> cv::Mat {
  cv::Mat visualization;
  cv::remap(frame, visualization, map_x, map_y, cv::INTER_LINEAR);

  for (std::size_t cluster_index = 0; cluster_index < clusters.size();
       ++cluster_index) {
    const gamepiece::kmeans_cluster_t& cluster = clusters.at(cluster_index);
    const auto pixel_cluster = PixelCluster(cluster, camera_matrix);
    const cv::Scalar color = ClusterColor(cluster.id);
    const cv::Vec3b pixel_color{static_cast<uchar>(color[0]),
                                static_cast<uchar>(color[1]),
                                static_cast<uchar>(color[2])};

    for (const cv::Point2f& point : pixel_cluster.img_points) {
      const cv::Point pixel{static_cast<int>(std::lround(point.x)),
                            static_cast<int>(std::lround(point.y))};
      if (pixel.x >= 0 && pixel.x < visualization.cols && pixel.y >= 0 &&
          pixel.y < visualization.rows) {
        visualization.at<cv::Vec3b>(pixel) = pixel_color;
      }
    }

    const cv::RotatedRect ellipse =
        PixelEllipse(gamepiece::ClusterEllipse(cluster), camera_matrix);
    cv::ellipse(visualization, ellipse, cv::Scalar{0, 0, 0}, 6, cv::LINE_AA);
    cv::ellipse(visualization, ellipse, color, 2, cv::LINE_AA);

  }
  const double horizon_y = camera_matrix.at<double>(1, 2) -
                           std::tan(pitch) * camera_matrix.at<double>(1, 1);
  if (horizon_y >= 0 && horizon_y <= visualization.rows - 1) {
    const int y = static_cast<int>(std::lround(horizon_y));
    cv::line(visualization, {0, y}, {visualization.cols - 1, y},
             cv::Scalar{0, 0, 0}, 4, cv::LINE_AA);
    cv::line(visualization, {0, y}, {visualization.cols - 1, y},
             cv::Scalar{0, 255, 255}, 2, cv::LINE_AA);
  }
  // Keep distances in a legend so nearby cluster labels stay readable.
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    const auto& position = clusters[i].robot_relative_translation;
    const auto center =
        PixelEllipse(gamepiece::ClusterEllipse(clusters[i]), camera_matrix)
            .center;
    DrawClusterIndex(visualization, clusters[i].id,
                     {static_cast<int>(std::lround(center.x)),
                      static_cast<int>(std::lround(center.y))},
                     NearestClusterDistance(clusters, i));
    if (!position.has_value()) {
      continue;
    }
    const std::string label =
        cv::format("%zu: %.2f m", clusters[i].id, position->Norm().value());
    const cv::Point origin{20, 80 + static_cast<int>(i) * 28};
    cv::putText(visualization, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar{0, 0, 0}, 5, cv::LINE_AA);
    cv::putText(visualization, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar{255, 255, 255}, 2, cv::LINE_AA);
  }
  return visualization;
}

auto ReadJson(const std::filesystem::path& path) -> nlohmann::json {
  std::ifstream file(path);
  CHECK(file.is_open()) << "Cannot open calibration: " << path;
  nlohmann::json json;
  file >> json;
  return json;
}

}  // namespace

auto main(int argc, char** argv) -> int {
  std::vector<std::string> positional;
  std::string frames_directory(kDefaultFramesDirectory);
  std::string poses_path;
  std::string sim_run;
  std::string output_dir;
  int start_frame = -1;
  int end_frame = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg.starts_with("--sim_run=")) {
      sim_run = arg.substr(10);
    } else if (arg.starts_with("--output_dir=")) {
      output_dir = arg.substr(13);
    } else if (arg.starts_with("--poses=")) {
      poses_path = arg.substr(8);
    } else if (arg.starts_with("--start=")) {
      start_frame = std::stoi(std::string(arg.substr(8)));
    } else if (arg.starts_with("--end=")) {
      end_frame = std::stoi(std::string(arg.substr(6)));
    } else if (arg.starts_with("--frames_dir=")) {
      frames_directory = std::string(arg.substr(13));
    } else {
      positional.emplace_back(arg);
    }
  }
  const bool use_range = start_frame >= 0 || end_frame >= 0;
  const bool use_defaults = positional.empty();
  if (!sim_run.empty() &&
      (!positional.empty() || use_range || !poses_path.empty())) {
    std::cerr << "--sim_run supplies frames, calibration, and poses; use only "
                 "--output_dir with it\n";
    return 1;
  }
  if (use_range && (start_frame < 0 || end_frame < start_frame)) {
    std::cerr << "--start and --end must both be set with start <= end\n";
    return 1;
  }
  if (!use_defaults &&
      (positional.size() < 3 || (positional.size() == 3 && !use_range))) {
    std::cerr
        << "Usage: hsv_cluster_visualizer [--poses=poses.json] [--start=N --end=M "
           "[--frames_dir=DIR]] [<intrinsics.json> <extrinsics.json> "
           "<output-directory> <frame.jpg>...]\n"
           "Or: hsv_cluster_visualizer --sim_run=RUN_OR_MANIFEST "
           "[--output_dir=DIR]\n"
           "With no positional arguments, uses the default calibration and "
           "output; with no range either, runs the sample frame.\n"
           "Calibration uses nested camera JSON (degrees) or flat BOS JSON "
           "(radians).\n";
    return 1;
  }
  try {
    nlohmann::json sim;
    std::filesystem::path sim_directory;
    if (!sim_run.empty()) {
      const auto manifest = std::filesystem::is_directory(sim_run)
                                ? std::filesystem::path(sim_run) / "manifest.json"
                                : std::filesystem::path(sim_run);
      sim_directory = manifest.parent_path();
      sim = ReadJson(manifest);
      if (!sim.at("frames").is_array() || sim.at("frames").empty()) {
        throw std::runtime_error("Simulation manifest must have nonempty frames");
      }
    }
    const auto sim_camera = [&]() -> nlohmann::json {
      if (sim.contains("camera_calibration")) return sim.at("camera_calibration");
      std::filesystem::path path = sim.at("camera_config").get<std::string>();
      if (path.is_relative() && !std::filesystem::exists(path)) {
        path = sim_directory / path;
      }
      return ReadJson(path);
    };
    const auto intrinsics_json =
        !sim.is_null()
            ? sim_camera()
            : ReadJson(use_defaults ? std::filesystem::path{kDefaultIntrinsicsPath}
                                    : std::filesystem::path{positional[0]});
    const auto extrinsics_json =
        !sim.is_null()
            ? intrinsics_json
            : ReadJson(use_defaults ? std::filesystem::path{kDefaultExtrinsicsPath}
                                    : std::filesystem::path{positional[1]});
    const auto& intrinsics = intrinsics_json.contains("intrinsics")
                                 ? intrinsics_json.at("intrinsics")
                                 : intrinsics_json;
    const bool rotations_in_degrees = extrinsics_json.contains("extrinsics");
    const auto& extrinsics = rotations_in_degrees
                                 ? extrinsics_json.at("extrinsics")
                                 : extrinsics_json;
    const double rotation_scale =
        rotations_in_degrees ? std::numbers::pi / 180.0 : 1.0;
    const cv::Matx33d camera_matrix{intrinsics.at("fx").get<double>(),
                                    0,
                                    intrinsics.at("cx").get<double>(),
                                    0,
                                    intrinsics.at("fy").get<double>(),
                                    intrinsics.at("cy").get<double>(),
                                    0,
                                    0,
                                    1};
    cv::Vec<double, 5> distortion{
        intrinsics.value("k1", 0.0), intrinsics.value("k2", 0.0),
        intrinsics.value("p1", 0.0), intrinsics.value("p2", 0.0),
        intrinsics.value("k3", 0.0)};
    if (!sim.is_null() && !sim.value("apply_distortion", true)) distortion = {};
    const frc::Pose3d camera_pose(
        units::meter_t{extrinsics.at("translation_x").get<double>()},
        units::meter_t{extrinsics.at("translation_y").get<double>()},
        units::meter_t{extrinsics.at("translation_z").get<double>()},
        frc::Rotation3d(
            units::radian_t{extrinsics.at("rotation_x").get<double>() *
                            rotation_scale},
            units::radian_t{extrinsics.at("rotation_y").get<double>() *
                            rotation_scale},
            units::radian_t{extrinsics.at("rotation_z").get<double>() *
                            rotation_scale}));
    // Simulation and cached poses bypass AprilTag localization.
    auto poses = poses_path.empty() ? nlohmann::json::object()
                                    : ReadJson(poses_path);
    if (!sim.is_null()) {
      for (const auto& frame : sim.at("frames")) {
        const auto& pose = frame.at("robot_pose");
        const auto& translation = pose.at("translation_m");
        const auto rotation = pose.value(
            "rotation_rpy_rad",
            std::vector<double>{0, 0, pose.value("rotation_rad", 0.0)});
        if (rotation.size() != 3 || rotation[0] != 0 || rotation[1] != 0 ||
            (translation.size() == 3 && translation.at(2).get<double>() != 0)) {
          throw std::runtime_error("Simulation heatmaps require planar robot poses");
        }
        const auto filename =
            std::filesystem::path(frame.at("file").get<std::string>()).filename();
        poses[filename.string()] =
            {{"x", translation.at(0)}, {"y", translation.at(1)}, {"yaw", rotation[2]}};
      }
    }
    std::optional<localization::MultiTagSolverNode> tag_solver;
    if (poses.empty() && intrinsics_json.contains("intrinsics") &&
        rotations_in_degrees) {
      tag_solver.emplace(
          "tags", "pose",
          camera::Intrinsics{use_defaults
                                 ? std::filesystem::path{kDefaultIntrinsicsPath}
                                 : std::filesystem::path{positional[0]}},
          camera::Extrinsics{use_defaults
                                 ? std::filesystem::path{kDefaultExtrinsicsPath}
                                 : std::filesystem::path{positional[1]}});
    }
    std::optional<frc::AprilTagDetector> tag_detector;
    if (tag_solver) {
      tag_detector.emplace();
      CHECK(tag_detector->AddFamily("tag36h11"));
      frc::AprilTagDetector::Config config;
      config.quadDecimate = 1.0F;
      tag_detector->SetConfig(config);
    }
    std::optional<frc::Pose2d> robot_pose;  // Held across frames without tags.
    const std::filesystem::path output_directory(
        !output_dir.empty() ? std::filesystem::path(output_dir) :
        !sim.is_null() ? std::filesystem::path("/root/debug-output") /
                            sim_directory.parent_path().filename() /
                            sim_directory.filename() / "clusters" :
        use_defaults ? std::filesystem::path{kDefaultOutputDirectory}
                     : std::filesystem::path{positional[2]});
    std::filesystem::create_directories(output_directory);
    nlohmann::json frames = nlohmann::json::array();
    std::vector<std::filesystem::path> input_paths;
    if (!sim.is_null()) {
      for (const auto& frame : sim.at("frames")) {
        input_paths.push_back(sim_directory / frame.at("file").get<std::string>());
      }
    }
    for (std::size_t i = 3; i < positional.size(); ++i) {
      input_paths.emplace_back(positional[i]);
    }
    for (int frame_number = start_frame; use_range && frame_number <= end_frame;
         ++frame_number) {
      char name[32];
      std::snprintf(name, sizeof(name), "frame_%06d.jpg", frame_number);
      input_paths.push_back(std::filesystem::path{frames_directory} / name);
    }
    if (input_paths.empty()) {
      input_paths.emplace_back(kDefaultFramePath);
    }
    const cv::Mat first_frame =
        cv::imread(input_paths.front().string(), cv::IMREAD_COLOR);
    CHECK(!first_frame.empty())
        << "Cannot decode frame: " << input_paths.front();
    gamepiece::HSVClusterTrackerNode tracker(
        "decoded", "clusters", camera_matrix,
        utils::EigenToCvMat(camera_pose.ToMatrix()), distortion,
        first_frame.total());
    const std::size_t frame_count = input_paths.size();
    cv::Mat map_x, map_y;
    cv::initUndistortRectifyMap(camera_matrix, distortion, cv::Mat(),
                               camera_matrix, first_frame.size(), CV_16SC2,
                               map_x, map_y);
    const std::size_t workers = std::min<std::size_t>(6, frame_count);
    control_loop::ThreadPool pool(workers);
    std::deque<std::future<cv::Mat>> pending;
    // Evenly spaced subset of up to 5 frames for the comparison panel.
    const std::size_t panel_size = std::min<std::size_t>(frame_count, 5);
    std::set<std::size_t> panel_indices;
    for (std::size_t i = 0; i < panel_size; ++i) {
      panel_indices.insert(
          panel_size == 1 ? 0 : i * (frame_count - 1) / (panel_size - 1));
    }
    std::vector<cv::Mat> panel_frames;
    std::vector<std::filesystem::path> heatmap_panel_paths;
    const cv::Mat field_base = MakeFieldCanvas();
    cv::Mat field_overlay = field_base.clone();
    std::map<std::size_t, std::vector<std::pair<cv::RotatedRect, cv::Scalar>>>
        field_ellipses;
    cv::Mat1f overall_point_counts(field_base.size(), 0.0f);
    std::size_t heatmap_frames = 0;
    auto next_frame = Async(pool, [first_frame] { return first_frame; });
    for (std::size_t frame_index = 0; frame_index < frame_count;
         ++frame_index) {
      const std::filesystem::path& input_path = input_paths[frame_index];
      const cv::Mat frame = next_frame.get();
      if (frame_index + 1 < frame_count) {
        next_frame = Async(pool, [path = input_paths[frame_index + 1]] {
          return cv::imread(path.string(), cv::IMREAD_COLOR);
        });
      }
      CHECK(!frame.empty()) << "Cannot decode frame: " << input_path;
      CHECK(frame.size() == first_frame.size())
          << "Frame dimensions changed: " << input_path;
      tracker.ProcessFrame(frame);
      if (frame_count == 1) {
        // Match BOS's single-frame visualizer by processing the frame twice.
        tracker.ProcessFrame(frame);
      }
      const auto& clusters = *tracker.GetClusters();
      const auto output_path =
          output_directory / (input_path.stem().string() + "_clusters.png");
      pending.push_back(Async(pool, [frame, clusters, camera_matrix, map_x, map_y,
                                    pitch = camera_pose.Rotation().Y().value(),
                                    input_path, output_path,
                                    panel = (use_range || !sim.is_null()) &&
                                            panel_indices.contains(frame_index)] {
        const cv::Mat visualization = AnnotateFrame(
            frame, clusters, cv::Mat(camera_matrix), map_x, map_y, pitch);
        const std::string caption = input_path.filename().string() + " | " +
                                    std::to_string(clusters.size()) + " clusters";
        cv::putText(visualization, caption, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar{0, 0, 0}, 5, cv::LINE_AA);
        cv::putText(visualization, caption, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar{255, 255, 255}, 2, cv::LINE_AA);
        CHECK(cv::imwrite(output_path.string(), visualization));
        cv::Mat thumbnail;
        if (panel) {
          cv::resize(visualization, thumbnail, {}, 0.5, 0.5, cv::INTER_AREA);
        }
        return thumbnail;
      }));
      std::size_t pose_tag_count = 0;
      std::vector<localization::tag_detection_t> detections;
      bool pose_updated = false;
      if (!poses.empty()) {
        // Missing entries are errors; null means no estimate, so hold the pose.
        const auto& pose = poses.at(input_path.filename().string());
        if (!pose.is_null()) {
          robot_pose.emplace(units::meter_t{pose.at("x").get<double>()},
                             units::meter_t{pose.at("y").get<double>()},
                             frc::Rotation2d{units::radian_t{
                                 pose.at("yaw").get<double>()}});
          pose_updated = true;
        }
      } else if (tag_solver) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        for (const auto* result : tag_detector->Detect(
                 gray.cols, gray.rows, static_cast<int>(gray.step), gray.data)) {
          localization::tag_detection_t detection{
              .tag_id = result->GetId(), .corners = {}};
          for (int corner = 0; corner < 4; ++corner) {
            detection.corners[corner] = {result->GetCorner(corner).x,
                                         result->GetCorner(corner).y};
          }
          detections.push_back(detection);
        }
        if (const auto estimate = tag_solver->AmbiguousSolve(detections)) {
          robot_pose = estimate->pos1.pose.ToPose2d();
          pose_tag_count = estimate->pos1.tag_ids.size();
          pose_updated = true;
        }
      }
      // Per-frame field view uses the camera view's cluster colors; the
      // all-frames overlay keeps one median-area ellipse per ID, colored by time.
      if (robot_pose.has_value()) {
        cv::Mat field = field_base.clone();
        const cv::Scalar frame_color =
            HueColor(static_cast<int>(frame_index * 120 / frame_count));
        for (const auto& cluster : clusters) {
          const cv::Scalar color = ClusterColor(cluster.id);
          if (const auto ellipse =
                  FieldEllipse(tracker, cluster, *robot_pose)) {
            DrawTranslucentEllipse(field, *ellipse, color);
            field_ellipses[cluster.id].emplace_back(*ellipse, frame_color);
          }
        }
        DrawRobot(field, *robot_pose, cv::Scalar{255, 255, 255});
        DrawRobot(field_overlay, *robot_pose, frame_color);
        std::vector<cv::Rect> field_labels;
        for (std::size_t i = 0; i < clusters.size(); ++i) {
          const auto& position = clusters[i].robot_relative_translation;
          if (!position.has_value()) {
            continue;
          }
          const auto center = robot_pose->Translation() +
                              position->RotateBy(robot_pose->Rotation());
          const auto pixel = FieldPoint(center.X().value(), center.Y().value());
          DrawClusterIndex(field, clusters[i].id,
                           {static_cast<int>(std::lround(pixel.x)),
                            static_cast<int>(std::lround(pixel.y))},
                           NearestClusterDistance(clusters, i), &field_labels);
        }
        const std::string field_caption =
            input_path.filename().string() + " | " +
            (pose_tag_count > 0 ? std::to_string(pose_tag_count) + " tags"
                               : pose_updated ? (sim.is_null() ? "cached pose"
                                                               : "simulated pose")
                                              : "no tags, pose held");
        const cv::Mat1f point_counts =
            ProjectedPointCounts(tracker, clusters, *robot_pose, field_base.size());
        overall_point_counts += point_counts;
        ++heatmap_frames;
        const auto heatmap_path = output_directory /
                                  (input_path.stem().string() + "_heatmap.png");
        if (panel_indices.contains(frame_index)) {
          heatmap_panel_paths.push_back(heatmap_path);
        }
        pending.push_back(Async(pool, [field_base, point_counts, field_caption,
                                      pose = *robot_pose, path = heatmap_path] {
          cv::Mat heatmap = PointHeatmap(field_base, point_counts, field_caption);
          DrawRobot(heatmap, pose, cv::Scalar{255, 255, 255});
          CHECK(cv::imwrite(path.string(), heatmap));
          return cv::Mat{};
        }));
        cv::putText(field, field_caption, {20, 30}, cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar{255, 255, 255}, 2, cv::LINE_AA);
        pending.push_back(Async(pool, [field, path = output_directory /
                                          (input_path.stem().string() + "_field.png")] {
          CHECK(cv::imwrite(path.string(), field));
          return cv::Mat{};
        }));
      }
      while (pending.size() >= workers) {
        auto thumbnail = pending.front().get();
        if (!thumbnail.empty()) panel_frames.push_back(std::move(thumbnail));
        pending.pop_front();
      }
      nlohmann::json cluster_data = nlohmann::json::array();
      std::size_t pixel_count = 0;
      for (const auto& normalized_cluster : clusters) {
        const auto cluster =
            PixelCluster(normalized_cluster, cv::Mat(camera_matrix));
        const auto ellipse =
            PixelEllipse(gamepiece::ClusterEllipse(normalized_cluster),
                         cv::Mat(camera_matrix));
        pixel_count += cluster.img_points.size();
        cluster_data.push_back(
            {{"id", cluster.id},
             {"centroid_px", {cluster.centroid.x, cluster.centroid.y}},
             {"pixel_count", cluster.img_points.size()},
             {"ellipse_semi_axes_px",
              {ellipse.size.width / 2.0f, ellipse.size.height / 2.0f}},
             {"ellipse_angle_deg", ellipse.angle},
             {"ground_position_m",
              {cluster.robot_relative_translation->X().value(),
               cluster.robot_relative_translation->Y().value()}},
             {"covariance",
              {{cluster.covar(0, 0), cluster.covar(0, 1)},
               {cluster.covar(1, 0), cluster.covar(1, 1)}}}});
      }
      frames.push_back({{"input", input_path.string()},
                        {"output", output_path.string()},
                        {"clusters", cluster_data}});
      std::cout << output_path << ": " << clusters.size() << " clusters, "
                << pixel_count << " thresholded pixels, " << detections.size()
                << " tags detected, " << pose_tag_count << " used for pose\n";
    }
    for (auto& result : pending) {
      auto thumbnail = result.get();
      if (!thumbnail.empty()) panel_frames.push_back(std::move(thumbnail));
    }
    if (robot_pose.has_value()) {
      for (auto& [id, ellipses] : field_ellipses) {
        // Select an observed ellipse; even counts use the upper median.
        const auto median = ellipses.begin() + ellipses.size() / 2;
        std::nth_element(ellipses.begin(), median, ellipses.end(),
                         [](const auto& a, const auto& b) {
                           return a.first.size.area() < b.first.size.area();
                         });
        DrawTranslucentEllipse(field_overlay, median->first, median->second);
      }
      CHECK(cv::imwrite((output_directory / "field_overlay.png").string(),
                        field_overlay));
      CHECK(cv::imwrite((output_directory / "overall_heatmap.png").string(),
                        PointHeatmap(field_base, overall_point_counts,
                                     "Overall projected points | " +
                                         std::to_string(heatmap_frames) + " frames")));
    }
    if (!panel_frames.empty()) {
      cv::Mat panel;
      cv::hconcat(panel_frames, panel);
      CHECK(cv::imwrite((output_directory / "comparison_panel.png").string(),
                        panel));
    }
    if (!heatmap_panel_paths.empty()) {
      std::vector<cv::Mat> thumbnails;
      for (const auto& path : heatmap_panel_paths) {
        cv::Mat thumbnail;
        cv::resize(cv::imread(path.string()), thumbnail, {}, 0.5, 0.5, cv::INTER_AREA);
        thumbnails.push_back(std::move(thumbnail));
      }
      cv::Mat panel;
      cv::hconcat(thumbnails, panel);
      CHECK(cv::imwrite((output_directory / "heatmap_comparison_panel.png").string(),
                        panel));
    }
    std::ofstream summary(output_directory / "clusters.json");
    CHECK(summary.is_open());
    summary << frames.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
