#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "camera/camera.h"
#include "camera/camera_constants.h"
#include "libuvc/libuvc.h"

namespace camera {

struct UVCCameraConfig {
  UVCCameraConfig(const std::string& path);
  explicit UVCCameraConfig(const camera_constant_t& camera_constant);
  std::string name;       // For debugging
  std::string serial_id;  // Used to find which camera to use
  int height;
  int width;
  int fps;
  int max_payload_size = 3072;
  int max_frame_size = 2048589;
};

// Default
// bmHint: 0001
// bFormatIndex: 1
// bFrameIndex: 1
// dwFrameInterval: 83333
// wKeyFrameRate: 0
// wPFrameRate: 0
// wCompQuality: 0
// wCompWindowSize: 0
// wDelay: 0
// dwMaxVideoFrameSize: 2048589
// dwMaxPayloadTransferSize: 3072
// bInterfaceNumber: 1

auto JpegBufferFromUvcFrame(const uvc_frame_t& frame)
    -> std::shared_ptr<JpegBuffer>;
auto UvcFrameTimestampMicros(const uvc_frame_t& frame) -> unsigned long;

class UVCCameraNode : public ICamera {
 public:
  UVCCameraNode(const UVCCameraConfig& config);
  ~UVCCameraNode() override;
  void RegisterCallback(const CameraCallback& callback) override;
  void Start() override;
  void CallBack(uvc_frame_t* frame);  // This should not be used publicly

 private:
  std::string name_;
  uvc_context_t* context_;
  uvc_device_t* device_;
  uvc_device_handle_t* device_handle_;
  uvc_stream_ctrl_t ctrl_;
  std::vector<CameraCallback> callbacks_;
  std::atomic<bool> start_ = false;
};

}  // namespace camera
