#pragma once

#include <iostream>
#include <optional>

#include "cuda_runtime_api.h"

namespace cos_test::testing {

inline auto CudaDeviceCount() -> std::optional<int> {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    std::cerr << "WARNING: CUDA device discovery failed: "
              << cudaGetErrorString(status) << '\n';
    return std::nullopt;
  }
  if (device_count <= 0) {
    std::cerr << "WARNING: no CUDA-capable GPU is present\n";
    return std::nullopt;
  }
  return device_count;
}

}  // namespace cos_test::testing
