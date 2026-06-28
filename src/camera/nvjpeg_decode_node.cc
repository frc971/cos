#include "camera/nvjpeg_decode_node.h"
#include <nvbufsurface.h>
#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {

NvjpegDecodeNode::NvjpegDecodeNode(const std::string& name) {
  decoder_ = NvJPEGDecoder::createJPEGDecoder(name.c_str());
  decode_thread_ = std::jthread([this](const std::stop_token& stop_token) {
    std::function<void()> task;
    while (!stop_token.stop_requested()) {
      {
        std::unique_lock<std::timed_mutex> lock(mutex_);
        cv_.wait(lock, [this, stop_token] {
          return !tasks_.empty() || stop_token.stop_requested();
        });
        if (tasks_.empty()) {
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      if (stop_token.stop_requested()) {
        break;
      }
      task();
    }
  });
}

NvjpegDecodeNode::~NvjpegDecodeNode() {
  LOG(INFO) << "Destructing NvjpegDecodeNode";
  decode_thread_.request_stop();
  cv_.notify_one();
  delete decoder_;
}
void NvjpegDecodeNode::Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer,
                              control_loops::MetaDataList metadata,
                              std::shared_ptr<control_loops::Context> ctx) {
  if (metadata.empty()) {
    LOG(WARNING) << "NvjpegDecodeNode received empty metadata";
  }
  std::function<void()> task = [this, jpeg_buffer, metadata, ctx] {
    DecodeJpegBuffer(jpeg_buffer, metadata, ctx);
  };
  {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    tasks_.push(task);
    cv_.notify_one();
  }
}

void NvjpegDecodeNode::RegisterCallback(
    const std::function<void(std::shared_ptr<DecodedJpegNvBuffer>,
                             control_loops::MetaDataList metadata,
                             std::shared_ptr<control_loops::Context>)>&
        callback) {
  std::unique_lock<std::timed_mutex> lock(mutex_, std::chrono::milliseconds(3));
  callbacks_.push_back(callback);
}

void NvjpegDecodeNode::DecodeJpegBuffer(
    const std::shared_ptr<JpegBuffer>& jpeg_buffer,
    control_loops::MetaDataList metadata,
    std::shared_ptr<control_loops::Context> ctx) {
  if (jpeg_buffer == nullptr) {
    for (const auto& callback : callbacks_) {
      callback(nullptr, metadata, ctx);
    }
    return;
  }

  uint32_t pixfmt;
  uint32_t width;
  uint32_t height;
  NvBuffer* buffer = nullptr;

  CHECK(!decoder_->decodeToBuffer(
      &buffer, static_cast<unsigned char*>(jpeg_buffer->ptr()),
      jpeg_buffer->size(), &pixfmt, &width, &height));

  auto buffer_shared_ptr = std::make_shared<DecodedJpegNvBuffer>(buffer);

  for (size_t i = 0; i < callbacks_.size(); i++) {  // NOLINT
    callbacks_[i](buffer_shared_ptr, metadata, ctx);
  }
}

}  // namespace camera
