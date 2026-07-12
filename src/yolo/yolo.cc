#include "yolo/yolo.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace yolo {
namespace {

auto LoadEngineFile(const std::string& filename) -> std::vector<char> {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Engine file not found: " + filename);
  }
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

auto GetOutputSize(nvinfer1::ICudaEngine* engine) -> size_t {
  const nvinfer1::Dims output_shape =
      engine->getTensorShape(engine->getIOTensorName(1));
  size_t output_size = 1;
  for (int i = 0; i < output_shape.nbDims; i++) {
    output_size *= output_shape.d[i];
  }
  return output_size;
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      LOG(WARNING) << "TensorRT: " << msg;
    }
  }
};

}  // namespace

Yolo::Yolo(const std::string& model_path, bool color, bool verbose)
    : color_(color), verbose_(verbose) {
  Logger logger;
  const std::vector<char> engine_data = LoadEngineFile(model_path);

  runtime_ = nvinfer1::createInferRuntime(logger);
  CHECK(runtime_ != nullptr);

  engine_ =
      runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  CHECK(engine_ != nullptr);

  context_ = engine_->createExecutionContext();
  CHECK(context_ != nullptr);

  input_dims_ = engine_->getTensorShape(engine_->getIOTensorName(0));
  size_t input_size = 1;
  for (int i = 0; i < input_dims_.nbDims; ++i) {
    input_size *= input_dims_.d[i];
  }

  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&input_buffer_),
                      sizeof(float) * input_size),
           cudaSuccess);
  output_size_ = GetOutputSize(engine_);
  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&output_buffer_),
                      sizeof(float) * output_size_),
           cudaSuccess);

  CHECK_EQ(cudaStreamCreate(&inference_cuda_stream_), cudaSuccess);
}

Yolo::~Yolo() {
  delete context_;
  delete engine_;
  delete runtime_;
  if (output_buffer_ != nullptr) {
    cudaFree(output_buffer_);
  }
  if (input_buffer_ != nullptr) {
    cudaFree(input_buffer_);
  }
  if (inference_cuda_stream_ != nullptr) {
    cudaStreamDestroy(inference_cuda_stream_);
  }
}

void Yolo::PreprocessImage(const cv::Mat& img, float* gpu_input,
                           const nvinfer1::Dims64&) {
  cv::Mat input;
  if (color_) {
    cv::cvtColor(img, input, cv::COLOR_BGR2RGB);
  } else if (img.channels() == 1) {
    input = img;
  } else {
    cv::cvtColor(img, input, cv::COLOR_BGR2GRAY);
  }

  cv::cuda::GpuMat img_gpu;
  img_gpu.upload(input);

  const int orig_h = input.rows;
  const int orig_w = input.cols;
  const float scale = std::min(TARGET_SIZE / static_cast<float>(orig_h),
                               TARGET_SIZE / static_cast<float>(orig_w));

  const int new_w = std::round(orig_w * scale);
  const int new_h = std::round(orig_h * scale);
  const int dw = TARGET_SIZE - new_w;
  const int dh = TARGET_SIZE - new_h;
  const int top = static_cast<int>(std::round(dh / 2.0 - 0.1));
  const int bottom = static_cast<int>(std::round(dh / 2.0 + 0.1));
  const int left = static_cast<int>(std::round(dw / 2.0 - 0.1));
  const int right = static_cast<int>(std::round(dw / 2.0 + 0.1));

  cv::cuda::GpuMat resized;
  cv::cuda::resize(img_gpu, resized, cv::Size(new_w, new_h), 0, 0,
                   cv::INTER_LINEAR);

  cv::cuda::GpuMat padded;
  cv::cuda::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  cv::cuda::GpuMat normalized;
  padded.convertTo(normalized, color_ ? CV_32FC3 : CV_32FC1, 1.F / 255.F);

  constexpr int channel_size = TARGET_SIZE * TARGET_SIZE;
  if (color_) {
    std::vector<cv::cuda::GpuMat> chw(3);
    cv::cuda::split(normalized, chw);
    for (int i = 0; i < 3; i++) {
      CHECK_EQ(
          cudaMemcpy(gpu_input + i * channel_size, chw[i].data,
                     channel_size * sizeof(float), cudaMemcpyDeviceToDevice),
          cudaSuccess);
    }
  } else {
    CHECK_EQ(cudaMemcpy(gpu_input, normalized.data,
                        channel_size * sizeof(float), cudaMemcpyDeviceToDevice),
             cudaSuccess);
  }
}

auto Yolo::RunModel(const cv::Mat& frame) -> std::vector<float> {
  PreprocessImage(frame, input_buffer_, input_dims_);
  context_->setTensorAddress(engine_->getIOTensorName(0), input_buffer_);
  context_->setTensorAddress(engine_->getIOTensorName(1), output_buffer_);
  CHECK(context_->enqueueV3(inference_cuda_stream_));

  CHECK_EQ(cudaStreamSynchronize(inference_cuda_stream_), cudaSuccess);
  std::vector<float> output(output_size_);
  CHECK_EQ(cudaMemcpy(output.data(), output_buffer_,
                      output_size_ * sizeof(float), cudaMemcpyDeviceToHost),
           cudaSuccess);
  if (verbose_) {
    LOG(INFO) << "YOLO output size: " << output.size();
  }
  return output;
}

auto Yolo::Postprocess(int original_height, int original_width,
                       const std::vector<float>& results,
                       std::vector<cv::Rect>& bboxes,
                       std::vector<float>& confidences,
                       std::vector<int>& class_ids) -> std::vector<float> {
  const float scale =
      std::min(TARGET_SIZE / static_cast<float>(original_height),
               TARGET_SIZE / static_cast<float>(original_width));
  const int new_w = std::round(original_width * scale);
  const int new_h = std::round(original_height * scale);
  const float pad_left = (TARGET_SIZE - new_w) / 2.0F;
  const float pad_top = (TARGET_SIZE - new_h) / 2.0F;

  constexpr int kNmsOutputSize = 6;
  const size_t count =
      std::min({bboxes.size(), confidences.size(), class_ids.size(),
                results.size() / kNmsOutputSize});
  for (size_t i = 0; i < count; i++) {
    float x1 = results[i * kNmsOutputSize];
    float y1 = results[i * kNmsOutputSize + 1];
    float x2 = results[i * kNmsOutputSize + 2];
    float y2 = results[i * kNmsOutputSize + 3];
    const float confidence = results[i * kNmsOutputSize + 4];
    const int id = static_cast<int>(results[i * kNmsOutputSize + 5]);

    x1 = (x1 - pad_left) / scale;
    y1 = (y1 - pad_top) / scale;
    x2 = (x2 - pad_left) / scale;
    y2 = (y2 - pad_top) / scale;

    x1 = std::clamp(x1, 0.0F, static_cast<float>(original_width));
    y1 = std::clamp(y1, 0.0F, static_cast<float>(original_height));
    x2 = std::clamp(x2, 0.0F, static_cast<float>(original_width));
    y2 = std::clamp(y2, 0.0F, static_cast<float>(original_height));

    bboxes[i] = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                         static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
    confidences[i] = confidence;
    class_ids[i] = id;
  }
  return results;
}

}  // namespace yolo
