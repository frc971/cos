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
ABSL_FLAG(std::string, camera_config, "constants/second_bot/left_camera.json",
          "Camera configuration with intrinsics and extrinsics");
ABSL_FLAG(std::string, field_dir, "constants/field-cad",
          "AdvantageScope field model directory");
ABSL_FLAG(std::string, output_dir, "sim-output", "Directory for rendered PNGs");
ABSL_FLAG(bool, apply_distortion, true, "Apply camera lens distortion");
ABSL_FLAG(double, fuel_entropy, -1.0,
          "Render staged Fuel and a second layout at this entropy (0 to 1)");
ABSL_FLAG(uint32_t, fuel_seed, 0, "Seed for the entropic Fuel layout");

namespace {

auto ReadJson(const std::filesystem::path& path) -> nlohmann::json {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open " + path.string());
  return nlohmann::json::parse(input);
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
    if (layout_entropy >= 0.0) {
      renderer.SetFuelLayout(layout_entropy, absl::GetFlag(FLAGS_fuel_seed));
    }
    const auto layout_dir = name.empty() ? output_dir : output_dir / name;
    std::filesystem::create_directories(layout_dir);
    nlohmann::json manifest = {
      {"camera_config", camera_path},
      {"poses", poses_path},
      {"field_model", (std::filesystem::path(field_dir) / "model.glb").string()},
      {"resolution", {image_size.width, image_size.height}},
      {"frames", nlohmann::json::array()}};
    if (layout_entropy >= 0.0) {
      manifest["fuel_entropy"] = layout_entropy;
      manifest["fuel_seed"] = absl::GetFlag(FLAGS_fuel_seed);
      manifest["gamepieces"] = nlohmann::json::array();
      for (const auto& position : renderer.FuelPositions()) {
        manifest["gamepieces"].push_back(
            {{"type", "Fuel"}, {"translation_m", position}});
      }
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
      filename << "frame_" << std::setw(6) << std::setfill('0') << index << ".png";
      const auto path = layout_dir / filename.str();
      if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write " + path.string());
      }
      manifest["frames"].push_back({{"file", filename.str()},
                                     {"robot_pose", item}});
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
