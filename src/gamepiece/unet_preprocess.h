#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gamepiece {

enum class UnetInputMode { kGrayscale = 1, kRgb = 3 };

auto PreprocessUnetImage(const unsigned char* input, int width, int height,
                         size_t stride, UnetInputMode input_mode, float* output,
                         int output_width, int output_height,
                         cudaStream_t stream) -> cudaError_t;

}  // namespace gamepiece
