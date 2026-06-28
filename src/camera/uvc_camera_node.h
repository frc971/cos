#pragma once

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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

class JpegBuffer {
 public:
  explicit JpegBuffer(size_t size) : size_(size), ptr_(std::malloc(size)) {}
  auto ptr() -> void* const { return ptr_; }

  auto size() -> size_t const { return size_; }
  ~JpegBuffer() { std::free(ptr_); }

 private:
  size_t size_;
  void* ptr_;
};

class UVCCameraNode {
 public:
  UVCCameraNode(const UVCCameraConfig& config);
  ~UVCCameraNode();
  void RegisterCallback(const std::function<void(std::shared_ptr<JpegBuffer>,
                                                 double timestamp)>& callback);
  void Start();
  void CallBack(uvc_frame_t* frame);  // This should not be used publicly

 private:
  std::string name_;
  uvc_context_t* context_;
  uvc_device_t* device_;
  uvc_device_handle_t* device_handle_;
  uvc_stream_ctrl_t ctrl_;
  std::vector<
      std::function<void(std::shared_ptr<JpegBuffer>, double timestamp)>>
      callbacks_;
  std::atomic<bool> start_ = false;
};

}  // namespace camera
