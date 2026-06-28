#pragma once
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>
#include "apriltag/apriltag_detector.h"
#include "third_party/971apriltag/apriltag.h"

namespace apriltag {

// GPU-accelerated AprilTag detector node using Austin's 971 apriltag library.
// If given an image with too low or too high exposure, the program will crash
// with 'invalid device ordinal'.
class GPUApriltagDetectorNode : public IApriltagDetectorNode {
 public:
  GPUApriltagDetectorNode(
      uint image_width, uint image_height, const nlohmann::json& intrinsics,
      vision::ImageFormat image_format = vision::ImageFormat::MONO8);
  ~GPUApriltagDetectorNode() override;

  void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>,
                               control_loops::MetaDataList metadata,
                               std::shared_ptr<control_loops::Context>)>&
          callback) override;
  void Detect(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx) override;

 private:
  frc::apriltag::CameraMatrix camera_matrix_;
  frc::apriltag::DistCoeffs distortion_coefficients_;
  apriltag_detector_t* apriltag_detector_;
  std::unique_ptr<frc::apriltag::GpuDetector> gpu_detector_;
  std::vector<std::function<void(std::shared_ptr<std::vector<tag_detection_t>>,
                                 control_loops::MetaDataList metadata,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;
  const vision::ImageFormat image_format_;
};

}  // namespace apriltag
