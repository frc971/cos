#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "camera/jpeg_buffer.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "logging/log_registration.h"

namespace camera {

class DecodedImageBuffer final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(DecodedImageBuffer);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) + data.capacity() * sizeof(uint8_t);
  }

  int width = 0;
  int height = 0;
  size_t stride = 0;
  std::vector<uint8_t> data;
  LOG_FIELDS(DecodedImageBuffer, width, height, stride)
};

class CpuJpegDecodeNode final : public control_loop::INode {
 public:
  CpuJpegDecodeNode(std::string_view input_path, std::string_view output_path,
                    control_loop::ThreadPool& thread_pool);

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

 private:
  auto Decode(const JpegBuffer* jpeg_buffer) -> DecodedImageBuffer;

  std::string input_path_;
  std::string output_path_;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

}  // namespace camera
