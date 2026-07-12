#include "camera/opencv_decode_node.h"

#include <cstring>

#include <opencv2/imgcodecs.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {
namespace {

// Builds an NvBuffer whose plane 0 mirrors the layout NvjpegDecodeNode
// produces (a single 8-bit grayscale/luma plane), so downstream consumers
// (apriltag detection, gamepiece detection) work unmodified regardless of
// which decoder produced the frame.
auto MakeGrayNvBuffer(const cv::Mat& gray) -> NvBuffer* {
  NvBuffer::NvBufferPlaneFormat fmt{};
  fmt.width = static_cast<uint32_t>(gray.cols);
  fmt.height = static_cast<uint32_t>(gray.rows);
  fmt.bytesperpixel = 1;
  fmt.stride = fmt.width;
  fmt.sizeimage = fmt.stride * fmt.height;

  auto* buffer = new NvBuffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                              V4L2_MEMORY_USERPTR, 1, &fmt, 0);
  CHECK_EQ(buffer->allocateMemory(), 0);

  NvBuffer::NvBufferPlane& plane = buffer->planes[0];
  for (uint32_t row = 0; row < fmt.height; ++row) {
    std::memcpy(plane.data + (row * fmt.stride), gray.ptr(static_cast<int>(row)),
               fmt.width);
  }
  plane.bytesused = fmt.sizeimage;
  plane.length = fmt.sizeimage;

  return buffer;
}

}  // namespace

OpenCVDecodeNode::OpenCVDecodeNode(const std::string& name) : name_(name) {
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
      // Release the task (and its captured Context) immediately so
      // Context::~Context() fires here, not on the next task's arrival --
      // otherwise WakeUp() never fires again once the queue drains, and
      // LoopController::Run() deadlocks waiting for a wakeup that only a new
      // task's arrival would trigger.
      task = nullptr;
    }
  });
}

OpenCVDecodeNode::~OpenCVDecodeNode() {
  LOG(INFO) << "Destructing OpenCVDecodeNode " << name_;
  decode_thread_.request_stop();
  cv_.notify_one();
}

void OpenCVDecodeNode::Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer,
                              control_loops::MetaDataList metadata,
                              std::shared_ptr<control_loops::Context> ctx) {
  if (metadata.empty()) {
    LOG(WARNING) << "OpenCVDecodeNode received empty metadata";
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

void OpenCVDecodeNode::RegisterCallback(
    const std::function<void(std::shared_ptr<DecodedJpegNvBuffer>,
                             control_loops::MetaDataList metadata,
                             std::shared_ptr<control_loops::Context>)>&
        callback) {
  std::unique_lock<std::timed_mutex> lock(mutex_, std::chrono::milliseconds(3));
  callbacks_.push_back(callback);
}

void OpenCVDecodeNode::DecodeJpegBuffer(
    const std::shared_ptr<JpegBuffer>& jpeg_buffer,
    control_loops::MetaDataList metadata,
    std::shared_ptr<control_loops::Context> ctx) {
  if (jpeg_buffer == nullptr) {
    for (const auto& callback : callbacks_) {
      callback(nullptr, metadata, ctx);
    }
    return;
  }

  const cv::Mat encoded(1, static_cast<int>(jpeg_buffer->size()), CV_8UC1,
                        jpeg_buffer->ptr());
  const cv::Mat gray = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
  CHECK(!gray.empty()) << "OpenCVDecodeNode failed to decode JPEG buffer";

  auto buffer_shared_ptr =
      std::make_shared<DecodedJpegNvBuffer>(MakeGrayNvBuffer(gray));

  for (const auto& callback : callbacks_) {
    callback(buffer_shared_ptr, metadata, ctx);
  }
}

}  // namespace camera
