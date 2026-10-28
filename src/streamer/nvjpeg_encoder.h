#pragma once

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

namespace streamer {
// JPEG encoder that avoids libjpeg: libnvmm_jpeg shadows its symbols in the robot
// binaries, which crashes cv::imencode. Not thread-safe.
class NvjpegEncoder {
 public:
  NvjpegEncoder() {
    Check(nvjpegCreateSimple(&handle_));
    Check(nvjpegEncoderStateCreate(handle_, &state_, nullptr));
    Check(nvjpegEncoderParamsCreate(handle_, &params_, nullptr));
    cudaStreamCreate(&stream_);
    Check(nvjpegEncoderParamsSetQuality(params_, 80, stream_));
    Check(nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, stream_));
  }
  NvjpegEncoder(const NvjpegEncoder&) = delete;
  auto operator=(const NvjpegEncoder&) -> NvjpegEncoder& = delete;
  ~NvjpegEncoder() {
    cudaFree(device_);
    cudaStreamDestroy(stream_);
    nvjpegEncoderParamsDestroy(params_);
    nvjpegEncoderStateDestroy(state_);
    nvjpegDestroy(handle_);
  }

  // Accepts CV_8UC1 or CV_8UC3 (BGR) images.
  auto Encode(const cv::Mat& image) -> std::string {
    cv::Mat bgr;
    if (image.channels() == 1) cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else bgr = image.isContinuous() ? image : image.clone();
    const size_t bytes = bgr.total() * 3;
    if (bytes > capacity_) {
      cudaFree(device_);
      if (cudaMalloc(&device_, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
      capacity_ = bytes;
    }
    cudaMemcpyAsync(device_, bgr.data, bytes, cudaMemcpyHostToDevice, stream_);
    nvjpegImage_t source{};
    source.channel[0] = static_cast<unsigned char*>(device_);
    source.pitch[0] = bgr.cols * 3;
    Check(nvjpegEncodeImage(handle_, state_, params_, &source, NVJPEG_INPUT_BGRI, bgr.cols, bgr.rows, stream_));
    size_t size = 0;
    Check(nvjpegEncodeRetrieveBitstream(handle_, state_, nullptr, &size, stream_));
    std::string jpeg(size, '\0');
    Check(nvjpegEncodeRetrieveBitstream(handle_, state_, reinterpret_cast<unsigned char*>(jpeg.data()), &size, stream_));
    cudaStreamSynchronize(stream_);
    return jpeg;
  }

 private:
  static void Check(nvjpegStatus_t status) {
    if (status != NVJPEG_STATUS_SUCCESS) throw std::runtime_error("nvjpeg encode failed: " + std::to_string(status));
  }
  nvjpegHandle_t handle_{};
  nvjpegEncoderState_t state_{};
  nvjpegEncoderParams_t params_{};
  cudaStream_t stream_{};
  void* device_ = nullptr;
  size_t capacity_ = 0;
};
}  // namespace streamer
