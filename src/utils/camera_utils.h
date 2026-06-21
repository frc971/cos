#pragma once
#include <string>
#include "nlohmann/json.hpp"

namespace utils {
auto ReadIntrinsics(const std::string& path) -> nlohmann::json;
auto ReadExtrinsics(const std::string& path) -> nlohmann::json;
}  // namespace utils
