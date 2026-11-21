#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <string>

namespace gamepiece {
// TensorRT ball segmentation network: 1x1xHxW grayscale in [0, 1], sigmoid probability out.
class UnetMask final : private nvinfer1::ILogger {
 public:
  explicit UnetMask(const std::string& engine_path);
  ~UnetMask();
  UnetMask(const UnetMask&) = delete;
  auto operator=(const UnetMask&) -> UnetMask& = delete;

  // Full-resolution 0/255 mask of pixels whose bilinearly upsampled probability >= confidence.
  void Threshold(const cv::Mat& image, float confidence, cv::Mat1b& mask);

 private:
  void log(Severity severity, const char* message) noexcept override;

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  float *input_device_ = nullptr, *output_device_ = nullptr;
  int height_ = 0, width_ = 0;
  cv::Mat gray_, small_, input_, probability_, full_;
};
}  // namespace gamepiece
