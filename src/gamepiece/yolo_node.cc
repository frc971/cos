#include "gamepiece/yolo_node.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {
namespace {
constexpr int kNmsOutputSize = 6;

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
  for (int i = 0; i < output_shape.nbDims; ++i) {
    output_size *= output_shape.d[i];
  }
  return output_size;
}
}  // namespace

YoloNode::YoloNode(std::string_view input_path, std::string_view output_path,
                   const std::string& model_path,
                   const std::vector<std::string>& class_names,
                   control_loop::ThreadPool& thread_pool)
    : input_path_(input_path),
      output_path_(output_path),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          std::vector<labeled_bounding_box_t>>(output_path_)}),
      class_names_(class_names) {
  const std::vector<char> engine_data = LoadEngineFile(model_path);

  runtime_ = nvinfer1::createInferRuntime(*this);
  CHECK(runtime_ != nullptr);
  engine_ =
      runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  CHECK(engine_ != nullptr);
  context_ = engine_->createExecutionContext();
  CHECK(context_ != nullptr);

  const nvinfer1::Dims64 input_dims =
      engine_->getTensorShape(engine_->getIOTensorName(0));
  CHECK_EQ(input_dims.nbDims, 4);
  CHECK_EQ(input_dims.d[0], 1);
  CHECK(input_dims.d[1] == 1 || input_dims.d[1] == 3);
  CHECK_GT(input_dims.d[2], 0);
  CHECK_EQ(input_dims.d[2], input_dims.d[3]);
  CHECK(engine_->getTensorDataType(engine_->getIOTensorName(0)) ==
        nvinfer1::DataType::kFLOAT);
  CHECK(engine_->getTensorDataType(engine_->getIOTensorName(1)) ==
        nvinfer1::DataType::kFLOAT);
  const auto output_dims = engine_->getTensorShape(engine_->getIOTensorName(1));
  CHECK_EQ(output_dims.nbDims, 3);
  CHECK_EQ(output_dims.d[0], 1);
  CHECK_EQ(output_dims.d[2], kNmsOutputSize);
  input_channels_ = input_dims.d[1];
  target_size_ = input_dims.d[2];
  size_t input_size = 1;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    input_size *= input_dims.d[i];
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

YoloNode::~YoloNode() {
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

void YoloNode::log(Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) {
    LOG(WARNING) << "TensorRT: " << message;
  }
}

auto YoloNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      context->SetMessage(output_path_, nullptr);
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }
    thread_pool_.Submit(
        [this, context, frame]() -> void {
          CHECK(frame->output_format == NVJPEG_OUTPUT_Y ||
                frame->output_format == NVJPEG_OUTPUT_BGRI);
          const int type =
              frame->output_format == NVJPEG_OUTPUT_Y ? CV_8UC1 : CV_8UC3;
          CHECK_GE(frame->stride,
                   static_cast<size_t>(frame->width * CV_MAT_CN(type)));
          const cv::cuda::GpuMat device(frame->height, frame->width, type,
                                        frame->destination.channel[0],
                                        frame->stride);
          auto detections = Detect(device);
          context->SetMessage(
              output_path_,
              std::make_shared<
                  control_loop::ValueMessage<bounding_box_detections_t>>(
                  std::move(detections)));
          for (const auto& callback : callbacks_) {
            callback(context);
          }
        },
        context->id);
  };
}

void YoloNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

auto YoloNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto YoloNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto YoloNode::Detect(const cv::cuda::GpuMat& image, float confidence)
    -> bounding_box_detections_t {
  if (!std::isfinite(confidence) || confidence < 0 || confidence > 1)
    throw std::invalid_argument("Confidence must be between zero and one");
  std::scoped_lock lock(detection_mutex_);
  auto detections = Postprocess(image.rows, image.cols, Run(image));
  std::erase_if(detections, [confidence](const auto& d) {
    return d.confidence < confidence;
  });
  return detections;
}

auto YoloNode::Run(const cv::cuda::GpuMat& image) -> std::vector<float> {
  Preprocess(image);
  context_->setTensorAddress(engine_->getIOTensorName(0), input_buffer_);
  context_->setTensorAddress(engine_->getIOTensorName(1), output_buffer_);
  CHECK(context_->enqueueV3(inference_cuda_stream_));

  CHECK_EQ(cudaStreamSynchronize(inference_cuda_stream_), cudaSuccess);
  std::vector<float> output(output_size_);
  CHECK_EQ(cudaMemcpy(output.data(), output_buffer_,
                      output_size_ * sizeof(float), cudaMemcpyDeviceToHost),
           cudaSuccess);
  return output;
}

auto YoloNode::Postprocess(int original_height, int original_width,
                           const std::vector<float>& results) const
    -> std::vector<labeled_bounding_box_t> {
  const float scale =
      std::min(target_size_ / static_cast<float>(original_height),
               target_size_ / static_cast<float>(original_width));
  const int new_width = std::round(original_width * scale);
  const int new_height = std::round(original_height * scale);
  const float pad_left = static_cast<float>((target_size_ - new_width) / 2);
  const float pad_top = static_cast<float>((target_size_ - new_height) / 2);

  std::vector<labeled_bounding_box_t> detections;
  const size_t count = results.size() / kNmsOutputSize;
  detections.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(results[i * kNmsOutputSize + 4]) ||
        results[i * kNmsOutputSize + 4] <= 0)
      continue;
    if (!std::all_of(results.begin() + i * kNmsOutputSize,
                     results.begin() + (i + 1) * kNmsOutputSize,
                     [](float x) { return std::isfinite(x); }))
      continue;
    float x1 = (results[i * kNmsOutputSize] - pad_left) / scale;
    float y1 = (results[i * kNmsOutputSize + 1] - pad_top) / scale;
    float x2 = (results[i * kNmsOutputSize + 2] - pad_left) / scale;
    float y2 = (results[i * kNmsOutputSize + 3] - pad_top) / scale;

    x1 = std::clamp(x1, 0.0F, static_cast<float>(original_width));
    y1 = std::clamp(y1, 0.0F, static_cast<float>(original_height));
    x2 = std::clamp(x2, 0.0F, static_cast<float>(original_width));
    y2 = std::clamp(y2, 0.0F, static_cast<float>(original_height));

    cv::Rect bounds(static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
    if (bounds.empty()) {
      continue;
    }

    const int class_id = static_cast<int>(results[i * kNmsOutputSize + 5]);
    const std::string label =
        class_id >= 0 && static_cast<size_t>(class_id) < class_names_.size()
            ? class_names_[class_id]
            : std::string{};
    detections.push_back(labeled_bounding_box_t{
        .bounds = bounds,
        .label = label,
        .class_id = class_id,
        .confidence = results[i * kNmsOutputSize + 4],
    });
  }
  return detections;
}

void YoloNode::Preprocess(const cv::cuda::GpuMat& image) {
  CHECK(!image.empty());
  CHECK_EQ(image.depth(), CV_8U);
  CHECK(image.channels() == 1 || image.channels() == 3);
  auto stream = cv::cuda::StreamAccessor::wrapStream(inference_cuda_stream_);
  cv::cuda::GpuMat converted = image;
  if (input_channels_ == 1 && image.channels() == 3) {
    cv::cuda::cvtColor(image, converted_, cv::COLOR_BGR2GRAY, 0, stream);
    converted = converted_;
  } else if (input_channels_ == 3) {
    cv::cuda::cvtColor(
        image, converted_,
        image.channels() == 1 ? cv::COLOR_GRAY2RGB : cv::COLOR_BGR2RGB, 0,
        stream);
    converted = converted_;
  }
  const float scale = std::min(target_size_ / static_cast<float>(image.rows),
                               target_size_ / static_cast<float>(image.cols));
  const int width = std::round(image.cols * scale),
            height = std::round(image.rows * scale);
  const int left = (target_size_ - width) / 2,
            top = (target_size_ - height) / 2;
  padded_.create(target_size_, target_size_, converted.type());
  padded_.setTo(cv::Scalar::all(114), stream);
  auto content = padded_(cv::Rect{left, top, width, height});
  cv::cuda::resize(converted, content, {width, height}, 0, 0, cv::INTER_LINEAR,
                   stream);
  const size_t pixels = target_size_ * target_size_;
  cv::cuda::GpuMat planes[3];
  for (int channel = 0; channel < input_channels_; ++channel)
    planes[channel] = cv::cuda::GpuMat(target_size_, target_size_, CV_32FC1,
                                       input_buffer_ + channel * pixels);
  if (input_channels_ == 1) {
    padded_.convertTo(planes[0], CV_32F, 1.F / 255.F, 0, stream);
  } else {
    padded_.convertTo(normalized_, CV_32F, 1.F / 255.F, 0, stream);
    cv::cuda::split(normalized_, planes, stream);
  }
}

}  // namespace gamepiece
