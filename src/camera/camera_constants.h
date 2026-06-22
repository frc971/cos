#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace camera {

enum class DetectorType { OPENCV_CPU, AUSTIN_GPU, INVALID };
enum class CameraType { UVC, MIPI, OPENCV, INVALID };

using camera_constant_t = struct CameraConstant {
  std::string name;
  std::optional<std::string> pipeline = std::nullopt;
  std::optional<std::string> intrinsics_path = std::nullopt;
  std::optional<std::string> extrinsics_path = std::nullopt;
  std::optional<double> backlight = std::nullopt;
  std::optional<uint> frame_width = std::nullopt;
  std::optional<uint> frame_height = std::nullopt;
  std::optional<uint> fps = std::nullopt;
  std::optional<double> exposure = std::nullopt;
  std::optional<double> brightness = std::nullopt;
  std::optional<double> sharpness = std::nullopt;
  std::optional<uint32_t> max_frame_size = std::nullopt;
  std::optional<uint32_t> max_payload_size = std::nullopt;
  std::optional<std::string> serial_id = std::nullopt;
  std::optional<double> stream_ratio = std::nullopt;
  std::optional<uint> port = std::nullopt;
  std::optional<uint> streamer_fps = std::nullopt;
  DetectorType detector_type = DetectorType::INVALID;
  CameraType camera_type = CameraType::INVALID;

  auto operator<(const CameraConstant& other) const -> bool {
    return name < other.name;
  }
};

using camera_constants_t = std::unordered_map<std::string, camera_constant_t>;

auto GetCameraConstants(const std::string& path) -> camera_constants_t;

auto GetCameraConstants() -> camera_constants_t;

}  // namespace camera
