#include "utils/camera_utils.h"
#include <fstream>
#include "absl/log/log.h"

namespace utils {

auto ReadIntrinsics(const std::string& path) -> nlohmann::json {
  nlohmann::json intrinsics;
  std::ifstream intrinsics_file(path);
  if (!intrinsics_file.is_open()) {
    LOG(FATAL) << "Error: Cannot open intrinsics file: " << path;
  } else {
    intrinsics_file >> intrinsics;
  }
  return intrinsics;
}

auto ReadExtrinsics(const std::string& path) -> nlohmann::json {
  nlohmann::json extrinsics;
  std::ifstream extrinsics_file(path);
  if (!extrinsics_file.is_open()) {
    LOG(FATAL) << "Error: Cannot open extrinsics file: " << path;
  } else {
    extrinsics_file >> extrinsics;
  }
  return extrinsics;
}

}  // namespace utils
