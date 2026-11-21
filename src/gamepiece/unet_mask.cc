#include "gamepiece/unet_mask.h"

#include <fstream>
#include <iterator>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace gamepiece {
UnetMask::UnetMask(const std::string& engine_path) {
  std::ifstream file(engine_path, std::ios::binary);
  if (!file) throw std::runtime_error("U-Net engine not found: " + engine_path);
  const std::vector<char> data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  runtime_ = nvinfer1::createInferRuntime(*this);
  CHECK(runtime_ != nullptr);
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) throw std::runtime_error("Cannot deserialize U-Net engine: " + engine_path);
  context_ = engine_->createExecutionContext();
  CHECK(context_ != nullptr);
  const auto in = engine_->getTensorShape(engine_->getIOTensorName(0));
  const auto out = engine_->getTensorShape(engine_->getIOTensorName(1));
  if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 1 || out.nbDims != 4 || out.d[2] != in.d[2] || out.d[3] != in.d[3])
    throw std::runtime_error("U-Net engine must map 1x1xHxW to 1x1xHxW");
  height_ = in.d[2], width_ = in.d[3];
  const size_t bytes = sizeof(float) * height_ * width_;
  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&input_device_), bytes), cudaSuccess);
  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&output_device_), bytes), cudaSuccess);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK(context_->setTensorAddress(engine_->getIOTensorName(0), input_device_));
  CHECK(context_->setTensorAddress(engine_->getIOTensorName(1), output_device_));
  probability_.create(height_, width_, CV_32F);
}

UnetMask::~UnetMask() {
  delete context_;
  delete engine_;
  delete runtime_;
  cudaFree(input_device_);
  cudaFree(output_device_);
  if (stream_) cudaStreamDestroy(stream_);
}

void UnetMask::log(Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) LOG(WARNING) << "TensorRT: " << message;
}

void UnetMask::Threshold(const cv::Mat& image, float confidence, cv::Mat1b& mask) {
  if (image.channels() == 3) cv::cvtColor(image, gray_, cv::COLOR_BGR2GRAY);
  else gray_ = image;
  cv::resize(gray_, small_, {width_, height_}, 0, 0, cv::INTER_AREA);
  small_.convertTo(input_, CV_32F, 1.0 / 255);
  const size_t bytes = sizeof(float) * height_ * width_;
  CHECK_EQ(cudaMemcpyAsync(input_device_, input_.data, bytes, cudaMemcpyHostToDevice, stream_), cudaSuccess);
  CHECK(context_->enqueueV3(stream_));
  CHECK_EQ(cudaMemcpyAsync(probability_.data, output_device_, bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  cv::resize(probability_, full_, image.size(), 0, 0, cv::INTER_LINEAR);
  cv::compare(full_, confidence, mask, cv::CMP_GE);
}
}  // namespace gamepiece
