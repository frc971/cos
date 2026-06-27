#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "NvJpegDecoder.h"

#include "camera/uvc_camera_node.h"
#include "utils/node.h"
namespace camera {

class DecodedJpegNvBuffer {
 public:
  ~DecodedJpegNvBuffer() { delete buffer; }
  NvBuffer* buffer;
};

class NvjpegDecodeNode : public INode<std::shared_ptr<DecodedJpegNvBuffer>> {
 public:
  NvjpegDecodeNode(const std::string& name);
  ~NvjpegDecodeNode() override;
  void RegisterCallback(
      const std::function<void(std::shared_ptr<DecodedJpegNvBuffer>)>&
          callback) override;
  void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  void DecodeJpegBuffer(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  NvJPEGDecoder* decoder_ = nullptr;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(std::shared_ptr<DecodedJpegNvBuffer>)>>
      callbacks_;
  std::jthread decode_thread_;
};

}  // namespace camera
