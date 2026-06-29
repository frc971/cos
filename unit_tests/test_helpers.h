#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "camera/camera_constants.h"
#include "nlohmann/json.hpp"

namespace cos_test::testing {

inline auto WriteJsonFile(const std::filesystem::path& path,
                          const nlohmann::json& json) -> std::string {
  std::ofstream out(path);
  out << json.dump(2);
  return path.string();
}

inline auto IntrinsicsJson() -> nlohmann::json {
  return {{"fx", 800.0}, {"fy", 800.0}, {"cx", 320.0}, {"cy", 240.0},
          {"k1", 0.0},   {"k2", 0.0},   {"p1", 0.0},   {"p2", 0.0},
          {"k3", 0.0}};
}

inline auto ExtrinsicsJson() -> nlohmann::json {
  return {{"translation_x", 0.0}, {"translation_y", 0.0},
          {"translation_z", 0.0}, {"rotation_x", 0.0},
          {"rotation_y", 0.0},    {"rotation_z", 0.0}};
}

inline auto MakeCameraConstant(const std::string& name,
                               const std::string& intrinsics_path,
                               const std::string& extrinsics_path)
    -> camera::camera_constant_t {
  camera::camera_constant_t camera_constant;
  camera_constant.name = name;
  camera_constant.serial_id = "serial-" + name;
  camera_constant.frame_width = 640;
  camera_constant.frame_height = 480;
  camera_constant.fps = 30;
  camera_constant.intrinsics_path = intrinsics_path;
  camera_constant.extrinsics_path = extrinsics_path;
  camera_constant.camera_type = camera::CameraType::UVC;
  camera_constant.detector_type = camera::DetectorType::OPENCV_CPU;
  return camera_constant;
}

}  // namespace cos_test::testing
