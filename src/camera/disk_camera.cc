#include "camera/disk_camera.h"

#include <cstring>
#include <fstream>
#include <iterator>

#include "absl/log/check.h"

namespace camera {
namespace {

auto ReadBytes(const std::filesystem::path& path)
    -> std::vector<unsigned char> {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

}  // namespace

DiskCamera::DiskCamera(std::filesystem::path frame_path,
                       unsigned long timestamp)
    : DiskCamera(ReadBytes(frame_path), timestamp) {}

DiskCamera::DiskCamera(std::vector<unsigned char> mjpeg_payload,
                       unsigned long timestamp)
    : mjpeg_payload_(std::move(mjpeg_payload)), timestamp_(timestamp) {
  CHECK(!mjpeg_payload_.empty());
}

void DiskCamera::RegisterCallback(const CameraCallback& callback) {
  callbacks_.push_back(callback);
}

void DiskCamera::Start() {
  auto buffer = std::make_shared<JpegBuffer>(mjpeg_payload_.size());
  std::memcpy(buffer->ptr(), mjpeg_payload_.data(), mjpeg_payload_.size());

  for (const auto& callback : callbacks_) {
    callback(buffer, timestamp_);
  }
}

}  // namespace camera
