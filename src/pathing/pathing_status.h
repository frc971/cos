#pragma once

#include <cstdint>

namespace pathing {

// These numeric values are part of the NetworkTables command wire format.
enum class PathingStatus : std::uint8_t {
  Inactive = 0,
  Following = 1,
  ReachedGoal = 2,
  InvalidInput = 3,
  StaleInput = 4,  // Reserved for command protocol compatibility.
  NoPath = 5,
  FitFailed = 6,
  Disconnected = 7,
};

}  // namespace pathing
