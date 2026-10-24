#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <units/angle.h>
#include <units/length.h>

#include "camera/camera_config.h"
#include "tools/path_camera_sim/field_renderer.h"

ABSL_FLAG(std::string, poses, "", "JSON file containing localization poses");
ABSL_FLAG(double, fps, 30.0, "Frame rate used for timestamp filenames in seconds");
ABSL_FLAG(std::string, camera_config, "constants/second_bot/left_camera.json",
          "Camera configuration with intrinsics and extrinsics");
ABSL_FLAG(std::string, field_dir, "constants/field-cad",
          "AdvantageScope field model directory");
ABSL_FLAG(std::string, output_dir, "/root/debug-output/path-camera-sim",
          "Directory for rendered PNGs");
ABSL_FLAG(bool, apply_distortion, true, "Apply camera lens distortion");
ABSL_FLAG(bool, fuel_truth, false, "Record occlusion-aware visible Fuel IDs for validation");
ABSL_FLAG(double, fuel_entropy, -1.0,
          "Render staged Fuel and a second layout at this entropy (0 to 1)");
ABSL_FLAG(uint32_t, fuel_seed, 0, "Seed for randomized Fuel layouts");
ABSL_FLAG(std::string, fuel_clusters, "",
          "JSON file containing an array of {center_m:[x,y], density:balls/m^2, "
          "covariance_m2:[v1,v2], rotation_rad:angle}; replaces staged Fuel");
ABSL_FLAG(double, fuel_noise, 0.0,
          "Additional isotropic position noise standard deviation in meters for --fuel_clusters");

namespace {

auto ReadJson(const std::filesystem::path& path) -> nlohmann::json {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open " + path.string());
  return nlohmann::json::parse(input);
}

auto ReadFuelClusters(const nlohmann::json& document)
    -> std::vector<path_camera_sim::FuelCluster> {
  if (!document.is_array()) throw std::runtime_error("--fuel_clusters file must contain a JSON array");
  std::vector<path_camera_sim::FuelCluster> clusters;
  for (const auto& item : document) {
    for (const auto* key : {"center_m", "covariance_m2"}) {
      if (!item.at(key).is_array() || item.at(key).size() != 2)
        throw std::runtime_error(std::string("Fuel cluster ") + key + " must have two values");
    }
    path_camera_sim::FuelCluster cluster{
        item.at("center_m").get<std::array<double, 2>>(),
        item.at("density").get<double>(),
        item.at("covariance_m2").get<std::array<double, 2>>(),
        item.at("rotation_rad").get<double>()};
    if (!std::isfinite(cluster.center_m[0]) || !std::isfinite(cluster.center_m[1]) ||
        !std::isfinite(cluster.density) || cluster.density < 0.0 ||
        !std::isfinite(cluster.covariance_m2[0]) || cluster.covariance_m2[0] <= 0.0 ||
        !std::isfinite(cluster.covariance_m2[1]) || cluster.covariance_m2[1] <= 0.0 ||
        !std::isfinite(cluster.rotation_rad))
      throw std::runtime_error("Fuel clusters require finite values, nonnegative density, and positive covariances");
    clusters.push_back(cluster);
  }
  return clusters;
}

auto ReadPose(const nlohmann::json& item) -> frc::Pose3d {
  const auto& translation = item.at("translation_m");
  if (!translation.is_array() ||
      (translation.size() != 2 && translation.size() != 3)) {
    throw std::runtime_error("translation_m must have two or three values");
  }
  const double x = translation.at(0).get<double>();
  const double y = translation.at(1).get<double>();
  const double z = translation.size() == 3 ? translation.at(2).get<double>() : 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  if (item.contains("rotation_rpy_rad")) {
    const auto& rotation = item.at("rotation_rpy_rad");
    if (!rotation.is_array() || rotation.size() != 3) {
      throw std::runtime_error("rotation_rpy_rad must have three values");
    }
    roll = rotation.at(0).get<double>();
    pitch = rotation.at(1).get<double>();
    yaw = rotation.at(2).get<double>();
  } else {
    yaw = item.at("rotation_rad").get<double>();
  }
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
      !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw)) {
    throw std::runtime_error("Pose contains a non-finite value");
  }
  return frc::Pose3d(units::meter_t{x}, units::meter_t{y}, units::meter_t{z},
                     frc::Rotation3d(units::radian_t{roll},
                                     units::radian_t{pitch},
                                     units::radian_t{yaw}));
}

auto CameraWorldToOpenCv(const frc::Pose3d& robot_pose,
                         const frc::Transform3d& camera_to_robot)
    -> Eigen::Matrix4d {
  const frc::Pose3d camera_pose =
      robot_pose.TransformBy(camera_to_robot.Inverse());
  Eigen::Matrix4d wpilib_to_opencv = Eigen::Matrix4d::Zero();
  wpilib_to_opencv(0, 1) = -1.0;
  wpilib_to_opencv(1, 2) = -1.0;
  wpilib_to_opencv(2, 0) = 1.0;
  wpilib_to_opencv(3, 3) = 1.0;
  return wpilib_to_opencv * camera_pose.ToMatrix().inverse();
}

void BuildDistortionMaps(const cv::Size& size, const cv::Matx33d& intrinsics,
                         const cv::Vec<double, 5>& distortion,
                         cv::Mat& map_x, cv::Mat& map_y) {
  std::vector<cv::Point2f> distorted_pixels;
  distorted_pixels.reserve(static_cast<size_t>(size.area()));
  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      distorted_pixels.emplace_back(x, y);
    }
  }
  std::vector<cv::Point2f> undistorted_pixels;
  cv::undistortPoints(distorted_pixels, undistorted_pixels, intrinsics,
                      distortion, cv::noArray(), intrinsics);
  map_x.create(size, CV_32FC1);
  map_y.create(size, CV_32FC1);
  for (int y = 0; y < size.height; ++y) {
    for (int x = 0; x < size.width; ++x) {
      const cv::Point2f& source = undistorted_pixels[y * size.width + x];
      map_x.at<float>(y, x) = source.x;
      map_y.at<float>(y, x) = source.y;
    }
  }
}

void Run() {
  const double fps = absl::GetFlag(FLAGS_fps);
  if (!std::isfinite(fps) || fps <= 0.0 || fps > 1e6)
    throw std::runtime_error("--fps must be finite and in (0, 1000000]");
  const std::string cluster_arg = absl::GetFlag(FLAGS_fuel_clusters);
  const auto cluster_json = cluster_arg.empty() ? nlohmann::json::array()
                                               : ReadJson(cluster_arg);
  const auto clusters = ReadFuelClusters(cluster_json);
  const double noise = absl::GetFlag(FLAGS_fuel_noise);
  if (!std::isfinite(noise) || noise < 0.0 || (cluster_arg.empty() && noise != 0.0))
    throw std::runtime_error("--fuel_noise requires --fuel_clusters and must be finite and nonnegative");
  if (!cluster_arg.empty() && absl::GetFlag(FLAGS_fuel_entropy) != -1.0)
    throw std::runtime_error("--fuel_clusters cannot be combined with --fuel_entropy");
  const std::string poses_path = absl::GetFlag(FLAGS_poses);
  if (poses_path.empty()) throw std::runtime_error("--poses is required");
  const auto camera_path = absl::GetFlag(FLAGS_camera_config);
  const camera::Intrinsics intrinsics(camera_path);
  const camera::Extrinsics extrinsics(camera_path);
  const auto camera_json = ReadJson(camera_path);
  const cv::Size image_size(camera_json.at("width").get<int>(),
                            camera_json.at("height").get<int>());
  if (image_size.width <= 0 || image_size.height <= 0) {
    throw std::runtime_error("Camera image dimensions must be positive");
  }
  const cv::Matx33d camera_matrix = intrinsics.ToMatrix();
  const auto camera_to_robot = extrinsics.ToCameraToRobot<frc::Transform3d>();
  const auto pose_document = ReadJson(poses_path);
  const auto& poses = pose_document.is_array() ? pose_document
                                               : pose_document.at("poses");
  if (!poses.is_array() || poses.empty()) {
    throw std::runtime_error("Pose file must contain a nonempty poses array");
  }
  cv::Mat map_x;
  cv::Mat map_y;
  if (absl::GetFlag(FLAGS_apply_distortion)) {
    BuildDistortionMaps(image_size, camera_matrix,
                        intrinsics.ToDistortionCoefficients(), map_x, map_y);
  }
  const auto field_dir = absl::GetFlag(FLAGS_field_dir);
  path_camera_sim::FieldRenderer renderer(field_dir, image_size, camera_matrix);
  const std::filesystem::path output_dir(absl::GetFlag(FLAGS_output_dir));
  const double entropy = absl::GetFlag(FLAGS_fuel_entropy);
  if (entropy != -1.0 && (!std::isfinite(entropy) || entropy < 0.0 || entropy > 1.0)) {
    throw std::runtime_error("--fuel_entropy must be between 0 and 1");
  }
  std::vector<std::pair<std::string, double>> layouts;
  if (entropy == -1.0) layouts.emplace_back("", -1.0);
  else {
    layouts.emplace_back("rigid", 0.0);
    layouts.emplace_back("entropic", entropy);
  }
  for (const auto& [name, layout_entropy] : layouts) {
    if (!cluster_arg.empty()) {
      renderer.SetFuelClusters(clusters, noise, absl::GetFlag(FLAGS_fuel_seed));
    }
    if (layout_entropy >= 0.0) {
      renderer.SetFuelLayout(layout_entropy, absl::GetFlag(FLAGS_fuel_seed));
    }
    const auto layout_dir = name.empty() ? output_dir : output_dir / name;
    std::filesystem::create_directories(layout_dir);
    nlohmann::json manifest = {
      {"camera_config", camera_path},
      {"camera_calibration", camera_json},
      {"apply_distortion", absl::GetFlag(FLAGS_apply_distortion)},
      {"poses", poses_path},
      {"field_model", (std::filesystem::path(field_dir) / "model.glb").string()},
      {"resolution", {image_size.width, image_size.height}},
      {"fps", fps},
      {"frames", nlohmann::json::array()}};
    if (!cluster_arg.empty()) {
      manifest["fuel_clusters"] = cluster_json;
      manifest["fuel_noise"] = noise;
    }
    if (layout_entropy >= 0.0) manifest["fuel_entropy"] = layout_entropy;
    if (layout_entropy >= 0.0 || !cluster_arg.empty()) {
      manifest["fuel_seed"] = absl::GetFlag(FLAGS_fuel_seed);
      manifest["gamepieces"] = nlohmann::json::array();
      for (std::size_t i = 0; i < renderer.FuelPositions().size(); ++i) {
        const auto& position = renderer.FuelPositions()[i];
        manifest["gamepieces"].push_back(
            {{"type", "Fuel"}, {"translation_m", position}});
        if (!cluster_arg.empty()) manifest["gamepieces"].back()["cluster_id"] = renderer.FuelClusterIds()[i];
      }
    }
    std::vector<std::array<double, 3>> centroids(clusters.size());
    std::vector<std::size_t> counts(clusters.size());
    if (!cluster_arg.empty()) {
      for (std::size_t i = 0; i < renderer.FuelPositions().size(); ++i) {
        const auto group = renderer.FuelClusterIds()[i];
        ++counts[group];
        for (int axis = 0; axis < 3; ++axis) centroids[group][axis] += renderer.FuelPositions()[i][axis];
      }
      for (std::size_t g = 0; g < centroids.size(); ++g)
        if (counts[g]) for (auto& coordinate : centroids[g]) coordinate /= counts[g];
      manifest["cluster_centroids_m"] = centroids;
    }
    for (size_t index = 0; index < poses.size(); ++index) {
      const auto& item = poses.at(index);
      const frc::Pose3d pose = ReadPose(item);
      cv::Mat image = renderer.Render(CameraWorldToOpenCv(pose, camera_to_robot));
      if (!map_x.empty()) {
        cv::Mat distorted;
        cv::remap(image, distorted, map_x, map_y, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
        image = std::move(distorted);
      }
      std::ostringstream filename;
      const double timestamp = static_cast<double>(index) / fps;
      filename << std::fixed << std::setprecision(6) << timestamp << ".png";
      const auto path = layout_dir / filename.str();
      if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write " + path.string());
      }
      manifest["frames"].push_back({{"file", filename.str()},
                                     {"timestamp", timestamp},
                                     {"robot_pose", item}});
      if (absl::GetFlag(FLAGS_fuel_truth)) {
        cv::Mat labels = renderer.Render(CameraWorldToOpenCv(pose, camera_to_robot), true);
        if (!map_x.empty()) cv::remap(labels.clone(), labels, map_x, map_y, cv::INTER_NEAREST);
        std::vector<std::size_t> visible(renderer.FuelPositions().size(), 0);
        for (int y = 0; y < labels.rows; ++y) for (int x = 0; x < labels.cols; ++x) {
          const auto p = labels.at<cv::Vec3b>(y, x);
          const std::size_t id = p[2] + (p[1] << 8) + (p[0] << 16);
          if (id > 0 && id <= visible.size()) ++visible[id - 1];
        }
        manifest["frames"].back()["visible_fuel_pixels"] = visible;
        if (!cluster_arg.empty()) {
          const auto camera = CameraWorldToOpenCv(pose, camera_to_robot);
          const auto probes = renderer.Render(camera, true, &centroids);
          std::vector<bool> centroid_visible(centroids.size(), false);
          for (int y = 0; y < probes.rows; ++y) for (int x = 0; x < probes.cols; ++x) {
            const auto p = probes.at<cv::Vec3b>(y, x);
            const std::size_t id = p[2] + (p[1] << 8) + (p[0] << 16);
            if (id > 0 && id <= centroid_visible.size()) centroid_visible[id - 1] = true;
          }
          // The point pass tests the undistorted frustum and opaque field depth;
          // also require its distorted projection to remain inside the output.
          for (std::size_t g = 0; g < centroids.size(); ++g) {
            if (!counts[g]) { centroid_visible[g] = false; continue; }
            const auto& c = centroids[g];
            const Eigen::Vector4d point = camera * Eigen::Vector4d(c[0], c[1], c[2], 1);
            std::vector<cv::Point2d> pixel;
            cv::projectPoints(std::vector<cv::Point3d>{{point[0], point[1], point[2]}},
                cv::Vec3d{}, cv::Vec3d{}, camera_matrix,
                map_x.empty() ? cv::Vec<double, 5>{} : intrinsics.ToDistortionCoefficients(), pixel);
            centroid_visible[g] = centroid_visible[g] && counts[g] &&
                pixel[0].x >= 0 && pixel[0].x < image_size.width &&
                pixel[0].y >= 0 && pixel[0].y < image_size.height;
          }
          manifest["frames"].back()["visible_cluster_centroids"] = centroid_visible;
        }
      }
    }
    std::ofstream output(layout_dir / "manifest.json");
    if (!output) throw std::runtime_error("Cannot write manifest.json");
    output << std::setw(2) << manifest << '\n';
    std::cout << "Rendered " << poses.size() << " "
              << (name.empty() ? "field" : name) << " frames to " << layout_dir << '\n';
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "path_camera_sim: " << error.what() << '\n';
    return 1;
  }
}
