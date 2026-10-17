#include "gamepiece/hsv_cluster_tracker_node.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "utils/cv_geometry.h"

namespace {

constexpr std::string_view kDefaultIntrinsicsPath =
    "constants/gamepiece/camera.json";
constexpr std::string_view kDefaultExtrinsicsPath =
    "constants/gamepiece/camera.json";
constexpr std::string_view kDefaultOutputDirectory =
    "build/fake_gamepiece-clusters";
constexpr std::string_view kDefaultFramesDirectory =
    "/cos-logs/downloaded_video_frames";
constexpr std::string_view kDefaultFramePath =
    "/cos-logs/downloaded_video_frames/frame_001500.jpg";

// Drawing helpers are adapted from BOS's hsv_cluster_visualizer.cc.
auto RandomClusterColor(std::mt19937& random_generator) -> cv::Scalar {
  std::uniform_int_distribution<int> hue_distribution(0, 179);
  const cv::Mat hsv_pixel(
      1, 1, CV_8UC3,
      cv::Scalar{static_cast<double>(hue_distribution(random_generator)), 220.0,
                 255.0});
  cv::Mat bgr_pixel;
  cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
  const cv::Vec3b color = bgr_pixel.at<cv::Vec3b>(0, 0);
  return {static_cast<double>(color[0]), static_cast<double>(color[1]),
          static_cast<double>(color[2])};
}

auto PixelEllipse(const gamepiece::Ellipse& ellipse,
                  const cv::Mat& camera_matrix)
    -> cv::RotatedRect {
  // Transform the algorithm's actual ellipse, including its normalized radius
  // padding. Unequal fx/fy can change both its axis lengths and orientation.
  const cv::Matx33d intrinsics(camera_matrix);
  const double cosine = std::cos(ellipse.rotation);
  const double sine = std::sin(ellipse.rotation);
  const cv::Matx22d axes{
      intrinsics(0, 0) * ellipse.semi_axes.width * cosine,
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

auto DrawClusterIndex(cv::Mat& image, const std::size_t cluster_index,
                      const cv::Point& centroid) -> void {
  const std::string label = std::to_string(cluster_index);
  constexpr double kFontScale = 0.8;
  constexpr int kTextThickness = 2;
  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(
      label, cv::FONT_HERSHEY_SIMPLEX, kFontScale, kTextThickness, &baseline);
  const cv::Point text_origin{centroid.x - text_size.width / 2,
                              centroid.y + text_size.height / 2};

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
  // Evaluate into fresh storage: result initially shares cluster.covar, and
  // assigning a MatExpr directly to it can overwrite the normalized covariance.
  const cv::Mat pixel_covariance =
      cv::Mat(scale) * cluster.covar * cv::Mat(scale.t());
  gamepiece::kmeans_cluster_t result = cluster;
  result.centroid = pixel_point(cluster.centroid);
  result.covar = pixel_covariance;
  for (auto& point : result.img_points) {
    point = pixel_point(point);
  }
  return result;
}

auto AnnotateFrame(const cv::Mat& frame,
                   const std::vector<gamepiece::kmeans_cluster_t>& clusters,
                   const cv::Mat& camera_matrix,
                   const cv::Mat& distortion_coeffs) -> cv::Mat {
  cv::Mat visualization;
  cv::undistort(frame, visualization, camera_matrix, distortion_coeffs,
                camera_matrix);

  constexpr std::mt19937::result_type kClusterColorSeed = 0x4B4D4541;
  std::mt19937 random_generator(kClusterColorSeed);
  for (std::size_t cluster_index = 0; cluster_index < clusters.size();
       ++cluster_index) {
    const gamepiece::kmeans_cluster_t& cluster = clusters.at(cluster_index);
    const auto pixel_cluster = PixelCluster(cluster, camera_matrix);
    const cv::Scalar color = RandomClusterColor(random_generator);
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

    const cv::Point centroid{
        static_cast<int>(std::lround(pixel_cluster.centroid.x)),
        static_cast<int>(std::lround(pixel_cluster.centroid.y))};
    DrawClusterIndex(visualization, cluster_index, centroid);
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
  int start_frame = -1;
  int end_frame = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg.starts_with("--start=")) {
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
  if (use_range && (start_frame < 0 || end_frame < start_frame)) {
    std::cerr << "--start and --end must both be set with start <= end\n";
    return 1;
  }
  if (!use_defaults && (positional.size() < 3 ||
                        (positional.size() == 3 && !use_range))) {
    std::cerr
        << "Usage: hsv_cluster_visualizer [--start=N --end=M "
           "[--frames_dir=DIR]] [<intrinsics.json> <extrinsics.json> "
           "<output-directory> <frame.jpg>...]\n"
           "With no positional arguments, uses the default calibration and "
           "output; with no range either, runs the sample frame.\n"
           "Calibration uses nested camera JSON (degrees) or flat BOS JSON "
           "(radians).\n";
    return 1;
  }
  try {
    const auto intrinsics_json =
        ReadJson(use_defaults ? std::filesystem::path{kDefaultIntrinsicsPath}
                              : std::filesystem::path{argv[1]});
    const auto extrinsics_json =
        ReadJson(use_defaults ? std::filesystem::path{kDefaultExtrinsicsPath}
                              : std::filesystem::path{argv[2]});
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
    const cv::Vec<double, 5> distortion{
        intrinsics.value("k1", 0.0), intrinsics.value("k2", 0.0),
        intrinsics.value("p1", 0.0), intrinsics.value("p2", 0.0),
        intrinsics.value("k3", 0.0)};
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
    // The tracker rotates camera rays into robot coordinates and uses the
    // camera's height in that frame. It needs the camera pose, not its inverse.
    gamepiece::HSVClusterTrackerNode tracker(
        "decoded", "clusters", camera_matrix,
        utils::EigenToCvMat(camera_pose.ToMatrix()), distortion);
    const std::filesystem::path output_directory(
        use_defaults ? std::filesystem::path{kDefaultOutputDirectory}
                     : std::filesystem::path{argv[3]});
    std::filesystem::create_directories(output_directory);
    nlohmann::json frames = nlohmann::json::array();
    std::vector<std::filesystem::path> input_paths;
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
    const std::size_t frame_count = input_paths.size();
    // Evenly spaced subset of up to 5 frames for the comparison panel.
    const std::size_t panel_size = std::min<std::size_t>(frame_count, 5);
    std::set<std::size_t> panel_indices;
    for (std::size_t i = 0; i < panel_size; ++i) {
      panel_indices.insert(panel_size == 1 ? 0
                                           : i * (frame_count - 1) /
                                                 (panel_size - 1));
    }
    std::vector<cv::Mat> panel_frames;
    for (std::size_t frame_index = 0; frame_index < frame_count;
         ++frame_index) {
      const std::filesystem::path& input_path = input_paths[frame_index];
      const cv::Mat frame = cv::imread(input_path.string(), cv::IMREAD_COLOR);
      CHECK(!frame.empty()) << "Cannot decode frame: " << input_path;
      tracker.ProcessFrame(frame);
      if (frame_count == 1) {
        // Match BOS's single-frame visualizer by processing the frame twice.
        tracker.ProcessFrame(frame);
      }
      const auto& clusters = *tracker.GetClusters();
      const cv::Mat visualization = AnnotateFrame(
          frame, clusters, cv::Mat(camera_matrix), cv::Mat(distortion));
      const std::string caption = input_path.filename().string() + " | " +
                                  std::to_string(clusters.size()) + " clusters";
      cv::putText(visualization, caption, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                  0.9, cv::Scalar{0, 0, 0}, 5, cv::LINE_AA);
      cv::putText(visualization, caption, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                  0.9, cv::Scalar{255, 255, 255}, 2, cv::LINE_AA);
      const auto output_path =
          output_directory / (input_path.stem().string() + "_clusters.png");
      CHECK(cv::imwrite(output_path.string(), visualization));
      if (use_range && panel_indices.contains(frame_index)) {
        cv::Mat thumbnail;
        cv::resize(visualization, thumbnail, {}, 0.5, 0.5, cv::INTER_AREA);
        panel_frames.push_back(thumbnail);
      }
      nlohmann::json cluster_data = nlohmann::json::array();
      std::size_t pixel_count = 0;
      for (const auto& normalized_cluster : clusters) {
        const auto cluster =
            PixelCluster(normalized_cluster, cv::Mat(camera_matrix));
        const auto ellipse = PixelEllipse(
            gamepiece::ClusterEllipse(normalized_cluster), cv::Mat(camera_matrix));
        pixel_count += cluster.img_points.size();
        cluster_data.push_back(
            {{"centroid_px", {cluster.centroid.x, cluster.centroid.y}},
             {"pixel_count", cluster.img_points.size()},
             {"ellipse_semi_axes_px",
              {ellipse.size.width / 2.0f, ellipse.size.height / 2.0f}},
             {"ellipse_angle_deg", ellipse.angle},
             {"ground_position_m",
              {cluster.camera_relative_translation->X().value(),
               cluster.camera_relative_translation->Y().value()}},
             {"covariance",
              {{cluster.covar.at<float>(0, 0), cluster.covar.at<float>(0, 1)},
               {cluster.covar.at<float>(1, 0),
                cluster.covar.at<float>(1, 1)}}}});
      }
      frames.push_back({{"input", input_path.string()},
                        {"output", output_path.string()},
                        {"clusters", cluster_data}});
      std::cout << output_path << ": " << clusters.size() << " clusters, "
                << pixel_count << " thresholded pixels\n";
    }
    if (!panel_frames.empty()) {
      cv::Mat panel;
      cv::hconcat(panel_frames, panel);
      CHECK(cv::imwrite((output_directory / "comparison_panel.png").string(),
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
