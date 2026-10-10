#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "control_loop/node.h"
#include "control_loop/thread_pool.h"

namespace gamepiece {

using labeled_bounding_box_t = struct labeled_bounding_box_t {
  cv::Rect bounds;
  std::string label;
  int class_id;
  float confidence;
};

using bounding_box_detections_t = std::vector<labeled_bounding_box_t>;

class YoloNode final : public control_loop::INode, private nvinfer1::ILogger {
 public:
  YoloNode(std::string_view input_path, std::string_view output_path,
           const std::string& model_path,
           const std::vector<std::string>& class_names,
           control_loop::ThreadPool& thread_pool);
  ~YoloNode() override;

  YoloNode(const YoloNode&) = delete;
  auto operator=(const YoloNode&) -> YoloNode& = delete;

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  auto Detect(const cv::cuda::GpuMat& image)
      -> std::vector<labeled_bounding_box_t>;
  void log(Severity severity, const char* message) noexcept override;
  void Preprocess(const cv::cuda::GpuMat& image);
  auto Run(const cv::cuda::GpuMat& image) -> std::vector<float>;
  [[nodiscard]] auto Postprocess(int original_height, int original_width,
                                 const std::vector<float>& results) const
      -> std::vector<labeled_bounding_box_t>;

  std::string input_path_;
  std::string output_path_;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex detection_mutex_;
  const std::vector<std::string> class_names_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t inference_cuda_stream_ = nullptr;
  float* input_buffer_ = nullptr;
  float* output_buffer_ = nullptr;
  size_t output_size_ = 0;
};

}  // namespace gamepiece
