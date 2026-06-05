#pragma once

#include "nlohmann/json.hpp"

class GpuApriltagDetectorNode {
  GpuApriltagDetectorNode(int width, int height,
                          const nlohmann::json& intrinsics);
};
