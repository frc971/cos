#include "gamepiece/unet_preprocess.h"

#include <cuda_runtime.h>

namespace gamepiece {
namespace {

__global__ void RotateResizeNormalize(const unsigned char* input, int width,
                                      int height, size_t stride, float* output,
                                      int output_width, int output_height) {
  const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (output_x >= output_width || output_y >= output_height) return;

  // Match OpenCV's half-pixel resize coordinates. Logged and live frames use
  // the same raw sensor orientation as the U-Net training images; camera roll
  // belongs in the extrinsics used for floor projection, not preprocessing.
  const float source_x =
      (output_x + 0.5F) * width / static_cast<float>(output_width) - 0.5F;
  const float source_y =
      (output_y + 0.5F) * height / static_cast<float>(output_height) - 0.5F;
  const int x0 = max(0, min(width - 1, static_cast<int>(floorf(source_x))));
  const int y0 = max(0, min(height - 1, static_cast<int>(floorf(source_y))));
  const int x1 = min(width - 1, x0 + 1);
  const int y1 = min(height - 1, y0 + 1);
  const float x_fraction = source_x - floorf(source_x);
  const float y_fraction = source_y - floorf(source_y);

  const float top = input[y0 * stride + x0] * (1.0F - x_fraction) +
                    input[y0 * stride + x1] * x_fraction;
  const float bottom = input[y1 * stride + x0] * (1.0F - x_fraction) +
                       input[y1 * stride + x1] * x_fraction;
  output[output_y * output_width + output_x] =
      (top * (1.0F - y_fraction) + bottom * y_fraction) / 255.0F;
}

}  // namespace

auto PreprocessUnetImage(const unsigned char* input, int width, int height,
                         size_t stride, float* output, int output_width,
                         int output_height, cudaStream_t stream) -> cudaError_t {
  constexpr dim3 kBlock(16, 16);
  const dim3 grid((output_width + kBlock.x - 1) / kBlock.x,
                  (output_height + kBlock.y - 1) / kBlock.y);
  RotateResizeNormalize<<<grid, kBlock, 0, stream>>>(
      input, width, height, stride, output, output_width, output_height);
  return cudaGetLastError();
}

}  // namespace gamepiece
