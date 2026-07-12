#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace yolo {

class Yolo {
 public:
  Yolo(const std::string& model_path, bool color, bool verbose = false);
  ~Yolo();

  auto RunModel(const cv::Mat& frame) -> std::vector<float>;
  auto Postprocess(int original_height, int original_width,
                   const std::vector<float>& results,
                   std::vector<cv::Rect>& bboxes,
                   std::vector<float>& confidences,
                   std::vector<int>& class_ids) -> std::vector<float>;

  static constexpr int TARGET_SIZE = 640;

 private:
  void PreprocessImage(const cv::Mat& frame, float* gpu_input,
                       const nvinfer1::Dims64& dims);

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t inference_cuda_stream_ = nullptr;
  nvinfer1::Dims64 input_dims_{};
  bool color_ = false;
  float* input_buffer_ = nullptr;
  float* output_buffer_ = nullptr;
  size_t output_size_ = 0;
  bool verbose_ = false;
};

}  // namespace yolo
