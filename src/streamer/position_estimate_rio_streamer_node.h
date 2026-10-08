#pragma once

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Pose3d.h>
#include <networktables/DoubleArrayTopic.h>
#include <networktables/StructTopic.h>
#include "camera/uvc_camera_node.h"
#include "control_loop/node.h"

namespace streamer {

class PositionEstimateRioStreamerNode final : public control_loop::INode {
 public:
  PositionEstimateRioStreamerNode(std::string_view input_path,
                                  std::string_view networktable_path);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  // We use the camera to get a average of the timestamps to report to the rio
  void AddCamera(const camera::UVCCameraNode& camera);

 private:
  auto GetTimestamp(const control_loop::Context& context) -> double;

 private:
  std::string input_path_;
  std::string networktable_path_;
  nt::NetworkTableInstance instance_;
  nt::DoubleArrayPublisher position_estimate_publisher_;
  nt::StructPublisher<frc::Pose2d> pose2d_publisher_;
  nt::StructPublisher<frc::Pose3d> pose3d_publisher_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<
      std::function<std::optional<double>(const control_loop::Context&)>>
      get_camera_timestamps_;
};

}  // namespace streamer
