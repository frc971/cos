#include "camera/camera_constants.h"

#include <fstream>
#include <optional>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "nlohmann/json.hpp"

ABSL_FLAG(std::string, camera_constants_path,           // NOLINT
          "/cos/constants/camera_constants.json",       // NOLINT
          "Path to the json file of camera constants");  // NOLINT

namespace camera {

auto GetCameraConstants() -> camera_constants_t {
  return GetCameraConstants(absl::GetFlag(FLAGS_camera_constants_path));
}

template <typename T>
void SetConstant(const std::string_view config_name, std::optional<T>& config,
                 const nlohmann::json& camera_config) {
  if (camera_config.contains(config_name) &&
      !camera_config[config_name].is_null()) {
    config = camera_config[config_name];
  }
}

auto StringToDetectorType(const std::string& detector_type) -> DetectorType {
  if (detector_type == "austin_gpu") {
    return DetectorType::AUSTIN_GPU;
  }
  if (detector_type == "opencv_cpu") {
    return DetectorType::OPENCV_CPU;
  }
  LOG(WARNING) << "Invalid detector type: " << detector_type;
  return DetectorType::INVALID;
}

auto StringToCameraType(const std::string& camera_type) -> CameraType {
  if (camera_type == "uvc") {
    return CameraType::UVC;
  }
  if (camera_type == "mipi") {
    return CameraType::MIPI;
  }
  if (camera_type == "opencv") {
    return CameraType::OPENCV;
  }
  LOG(WARNING) << "Invalid camera type: " << camera_type;
  return CameraType::INVALID;
}

auto GetCameraConstants(const std::string& path) -> camera_constants_t {
  camera_constants_t camera_constants;
  std::ifstream f(path);
  CHECK(f) << "Failed to read camera constants json: " << path;

  nlohmann::json json;
  f >> json;

  const nlohmann::json& camera_configs = json.at("cameras");
  for (const nlohmann::json& camera_config : camera_configs) {
    if (camera_config.is_null()) {
      LOG(WARNING) << "Found a null camera config";
      continue;
    }
    if (!camera_config.contains("name") || camera_config["name"].is_null()) {
      LOG(WARNING) << "Could not find name in camera config";
      continue;
    }
    if (camera_constants.contains(camera_config["name"])) {
      LOG(WARNING) << "Duplicate cameras";
      continue;
    }

    camera_constant_t camera_constant{
        .name = camera_config.value("name", std::string{})};

    SetConstant<std::string>("pipeline", camera_constant.pipeline,
                             camera_config);
    SetConstant<std::string>("intrinsics_path", camera_constant.intrinsics_path,
                             camera_config);
    SetConstant<std::string>("extrinsics_path", camera_constant.extrinsics_path,
                             camera_config);
    SetConstant<double>("backlight", camera_constant.backlight, camera_config);
    SetConstant<uint>("frame_width", camera_constant.frame_width,
                      camera_config);
    SetConstant<uint>("frame_height", camera_constant.frame_height,
                      camera_config);
    SetConstant<uint>("fps", camera_constant.fps, camera_config);
    SetConstant<double>("exposure", camera_constant.exposure, camera_config);
    SetConstant<double>("brightness", camera_constant.brightness,
                        camera_config);
    SetConstant<double>("sharpness", camera_constant.sharpness, camera_config);
    SetConstant<std::string>("serial_id", camera_constant.serial_id,
                             camera_config);
    SetConstant<uint32_t>("max_frame_size", camera_constant.max_frame_size,
                          camera_config);
    SetConstant<uint32_t>("max_payload_size", camera_constant.max_payload_size,
                          camera_config);
    SetConstant<double>("stream_ratio", camera_constant.stream_ratio,
                        camera_config);
    SetConstant<uint>("port", camera_constant.port, camera_config);
    SetConstant<uint>("streamer_fps", camera_constant.streamer_fps,
                      camera_config);

    if (camera_config.contains("detector_type") &&
        !camera_config["detector_type"].is_null()) {
      camera_constant.detector_type =
          StringToDetectorType(camera_config["detector_type"]);
    }

    if (camera_config.contains("camera_type") &&
        !camera_config["camera_type"].is_null()) {
      camera_constant.camera_type =
          StringToCameraType(camera_config["camera_type"]);
    }

    camera_constants.insert({camera_constant.name, camera_constant});
  }
  return camera_constants;
}

}  // namespace camera
