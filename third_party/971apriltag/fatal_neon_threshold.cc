#include "absl/log/log.h"

#include "threshold.h"

namespace frc::apriltag {

std::unique_ptr<Threshold> MakeNeonThreshold(
    vision::ImageFormat /*image_format*/, size_t /*width*/, size_t /*height*/) {
  LOG(FATAL) << "NEON threshold not implemented for this platform.";
}

}  // namespace frc::apriltag
