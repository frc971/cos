#pragma once

#include <filesystem>
#include <vector>

#include "camera/camera.h"

namespace camera {

class DiskCamera : public ICamera {
 public:
  DiskCamera(std::filesystem::path frame_path, unsigned long timestamp);
  DiskCamera(std::vector<unsigned char> mjpeg_payload, unsigned long timestamp);

  void RegisterCallback(const CameraCallback& callback) override;
  void Start() override;

 private:
  std::vector<unsigned char> mjpeg_payload_;
  unsigned long timestamp_;
  std::vector<CameraCallback> callbacks_;
};

}  // namespace camera
