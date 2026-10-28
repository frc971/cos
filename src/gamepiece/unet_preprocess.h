#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gamepiece {

auto PreprocessUnetImage(const unsigned char* input, int width, int height,
                         size_t stride, float* output, int output_width,
                         int output_height, cudaStream_t stream) -> cudaError_t;

}  // namespace gamepiece
