#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <string_view>
#include <vector>

#include <frc/geometry/struct/Pose2dStruct.h>
#include <networktables/NetworkTableInstance.h>
#include <networktables/StructTopic.h>

#include "control_loop/node.h"
#include "localization/pathing_input_message.h"

namespace localization {

// Reads the robot pose and goal from NetworkTables for the pathing loop.
class RobotPoseNode final : public control_loop::INode {
 public:
  static constexpr char kPose[] = "pathing:pose";
  static constexpr char kGoal[] = "pathing:goal";
  static constexpr char kInput[] = "pathing:input";

  RobotPoseNode(nt::NetworkTableInstance instance, std::string_view pose_topic,
                std::string_view goal_topic, std::chrono::milliseconds period);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) override;

 private:
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::mutex mutex_;
  nt::StructSubscriber<frc::Pose2d> pose_;
  nt::StructSubscriber<frc::Pose2d> goal_;
};

}  // namespace localization
