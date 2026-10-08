#pragma once
#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>

#include <nvjpeg.h>

#include "camera/jpeg_buffer.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "control_loop/timed_node.h"
#include "logging/log_registration.h"

namespace camera {

class DecodedJpegBuffer final : public control_loop::IMessage {
 public:
  ~DecodedJpegBuffer() override;
  DecodedJpegBuffer() = default;

  DecodedJpegBuffer(DecodedJpegBuffer&& other) noexcept;

  auto GetType() -> const std::type_info& override {
    return typeid(DecodedJpegBuffer);
  }
  auto GetSize() -> size_t override { return sizeof(*this) + output_size; }

  int width = 0;
  int height = 0;
  size_t stride = 0;
  size_t output_size = 0;
  nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_Y;
  std::array<size_t, NVJPEG_MAX_COMPONENT> channel_sizes = {};
  nvjpegImage_t destination = {};
  LOG_FIELDS(DecodedJpegBuffer, width, height, stride, output_size,
             output_format, channel_sizes)
};

class NvjpegDecodeNode final : public control_loop::INode,
                               public control_loop::ITimedNode {
 public:
  explicit NvjpegDecodeNode(std::string_view input_path,
                            std::string_view output_path,
                            nvjpegOutputFormat_t output_format,
                            control_loop::ThreadPool& thread_pool);
  ~NvjpegDecodeNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override {
    callbacks_.emplace_back(callback);
  };
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void EnableTiming(std::string_view latency_channel) override;

 private:
  auto DecodeJpegBuffer(const JpegBuffer* jpeg_buffer)
      -> std::optional<DecodedJpegBuffer>;

 private:
  std::string input_path_;
  std::string output_path_;
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegDecoder_t decoder_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  nvjpegJpegStream_t jpeg_stream_ = nullptr;
  nvjpegDecodeParams_t decode_params_ = nullptr;
  nvjpegBufferPinned_t pinned_buffer_ = nullptr;
  nvjpegBufferDevice_t device_buffer_ = nullptr;
  std::mutex decode_mutex_;
  nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_BGRI;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  cudaStream_t stream_ = nullptr;
  std::optional<std::string> latency_channel_ = std::nullopt;
};

}  // namespace camera
