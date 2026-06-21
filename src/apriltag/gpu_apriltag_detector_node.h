#pragma once
#include <functional>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "apriltag/apriltag_detector.h"
#include "third_party/971apriltag/apriltag.h"

namespace apriltag {

// GPU-accelerated AprilTag detector node using Austin's 971 apriltag library.
// If given an image with too low or too high exposure, the program will crash
// with 'invalid device ordinal'.
class GPUApriltagDetectorNode : public IApriltagDetectorNode {
 public:
  GPUApriltagDetectorNode(uint image_width, uint image_height,
                          const nlohmann::json& intrinsics);
  ~GPUApriltagDetectorNode() override;

  void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>&
          callback) override;
  void Detect(const camera::DecodedJpegNvBuffer& frame,
              double timestamp) override;

 private:
  static constexpr bool restart_detector_on_cuda_error = false;
  static constexpr vision::ImageFormat image_format =
      vision::ImageFormat::MONO8;
  frc::apriltag::CameraMatrix camera_matrix_;
  frc::apriltag::DistCoeffs distortion_coefficients_;
  apriltag_detector_t* apriltag_detector_;
  std::unique_ptr<frc::apriltag::GpuDetector> gpu_detector_;
  std::vector<std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>>
      callbacks_;
};

}  // namespace apriltag
