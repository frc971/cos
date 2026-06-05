#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "NvJpegDecoder.h"

#include "camera/uvc_camera_node.h"
namespace camera {

class NvjpegDecodeNode {
 public:
  NvjpegDecodeNode(const std::string& name);
  ~NvjpegDecodeNode();
  void RegisterCallback(const std::function<void(NvBuffer*)>& callback);
  void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  void DecodeJpegBuffer(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  NvJPEGDecoder* decoder_ = nullptr;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(NvBuffer*)>> callbacks_;
  std::jthread decode_thread_;
};

}  // namespace camera
