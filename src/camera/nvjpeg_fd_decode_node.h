#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "camera/jpeg_buffer.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "control_loop/timed_node.h"
#include "logging/log_registration.h"

struct CosNvjpegDecoder;

namespace camera {

class DecodedJpegFdBuffer final : public control_loop::IMessage {
 public:
  DecodedJpegFdBuffer() = default;
  ~DecodedJpegFdBuffer() override;
  DecodedJpegFdBuffer(DecodedJpegFdBuffer&& other) noexcept;

  auto GetType() -> const std::type_info& override {
    return typeid(DecodedJpegFdBuffer);
  }
  auto GetSize() -> size_t override { return sizeof(*this) + output_size; }

  int fd = -1;
  uint32_t pixel_format = 0;
  int width = 0;
  int height = 0;
  size_t stride = 0;
  size_t output_size = 0;
  LOG_FIELDS(DecodedJpegFdBuffer, fd, pixel_format, width, height, stride,
             output_size)
};

class NvjpegFdDecodeNode final : public control_loop::INode,
                                 public control_loop::ITimedNode {
 public:
  NvjpegFdDecodeNode(std::string_view input_path, std::string_view output_path,
                     control_loop::ThreadPool& thread_pool);
  ~NvjpegFdDecodeNode() override;

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override {
    callbacks_.emplace_back(callback);
  }
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void EnableTiming(std::string_view latency_channel) override;

 private:
  auto DecodeJpegBuffer(const JpegBuffer* jpeg_buffer)
      -> std::optional<DecodedJpegFdBuffer>;

  std::string input_path_;
  std::string output_path_;
  CosNvjpegDecoder* decoder_ = nullptr;
  std::mutex decode_mutex_;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::optional<std::string> latency_channel_ = std::nullopt;
};

}  // namespace camera
