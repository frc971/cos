#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "gamepiece/unet_preprocess.h"

namespace {

class UnetPreprocessTest
    : public testing::TestWithParam<gamepiece::UnetInputMode> {};

TEST_P(UnetPreprocessTest, MatchesCpuResizeWithPaddedRowsAndPlanarOutput) {
  const int channels = static_cast<int>(GetParam());
  for (const auto& sizes : std::vector<std::pair<cv::Size, cv::Size>>{
           {{3, 2}, {7, 5}}, {{7, 5}, {3, 2}}, {{3, 2}, {3, 2}},
           {{1, 1}, {4, 3}}}) {
    const auto [input_size, output_size] = sizes;
    SCOPED_TRACE(testing::Message() << input_size << " -> " << output_size);
    const size_t stride = input_size.width * channels + 11;
    std::vector<unsigned char> pixels(stride * input_size.height, 255);
    cv::Mat input(input_size, CV_MAKETYPE(CV_8U, channels), pixels.data(), stride);
    for (int y = 0; y < input.rows; ++y) {
      for (int x = 0; x < input.cols; ++x) {
        for (int c = 0; c < channels; ++c) {
          input.ptr<unsigned char>(y)[x * channels + c] =
              (x * 17 + y * 31 + c * 73) % 256;
        }
      }
    }
    cv::Mat normalized;
    input.convertTo(normalized, CV_32F, 1.0 / 255.0);
    cv::Mat expected;
    cv::resize(normalized, expected, output_size, 0, 0, cv::INTER_LINEAR);

    unsigned char* device_input = nullptr;
    float* device_output = nullptr;
    const size_t elements = output_size.area() * channels;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_input), pixels.size()),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_output),
                         elements * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input, pixels.data(), pixels.size(),
                         cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(gamepiece::PreprocessUnetImage(
                  device_input, input.cols, input.rows, stride, GetParam(),
                  device_output, output_size.width, output_size.height, nullptr),
              cudaSuccess);
    std::vector<float> actual(elements);
    EXPECT_EQ(cudaMemcpy(actual.data(), device_output, elements * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
    EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    for (int c = 0; c < channels; ++c) {
      for (int y = 0; y < expected.rows; ++y) {
        for (int x = 0; x < expected.cols; ++x) {
          EXPECT_NEAR(actual[c * output_size.area() + y * expected.cols + x],
                      expected.ptr<float>(y)[x * channels + c], 1e-6F);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    InputModes, UnetPreprocessTest,
    testing::Values(gamepiece::UnetInputMode::kGrayscale,
                    gamepiece::UnetInputMode::kRgb));

}  // namespace
