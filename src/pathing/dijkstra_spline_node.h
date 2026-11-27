#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "control_loop/node.h"
#include "pathing/dijkstra_planner.h"
#include "pathing/motion_settings.h"
#include "pathing/spline_path_message.h"

namespace pathing {

class CommandMessage;

// Footprint, search cost, and limits on spline refinement and sampling.
struct SplineSettings {
  double robot_radius_meters = 0;
  double turn_penalty = 0.3;  // meters/radian, as in MotionSettings
  size_t max_refinement_passes = 32;
  size_t max_control_points = 4096;
  units::meter_t sample_spacing{0.02};
};

struct SplineCommandSettings {
  MotionSettings motion;
  units::meter_t lookahead{0.5};
  std::chrono::milliseconds command_timeout{250};

  [[nodiscard]] auto IsValid() const -> bool;
};

// Fits a collision-checked spline and publishes the requested velocity.
class DijkstraSplineNode final : public control_loop::INode {
 public:
  DijkstraSplineNode(std::string_view input_channel,
                     std::string_view navgrid_channel,
                     std::string_view command_channel,
                     SplineSettings settings = {},
                     SplineCommandSettings command_settings = {});
  // Publish the fitted spline and an overlay with samples and waypoint markers.
  // Call before registering with the control loop.
  void EnableSplineDebug(std::string_view spline_channel,
                         std::string_view polyline_channel);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  auto Calculate(const frc::Pose2d& start, const frc::Pose2d& end,
                 const DijkstraPlanner& planner) const -> SplinePathMessage;
  void CalculateCommand(const frc::Pose2d& current, const frc::Pose2d& target,
                        const DijkstraPlanner& planner,
                        const SplinePathMessage& spline,
                        CommandMessage& command) const;

  std::string input_channel_;
  std::string navgrid_channel_;
  std::string command_channel_;
  std::string spline_debug_channel_;
  std::string polyline_debug_channel_;
  std::shared_ptr<const streamer::NavgridMessage> navgrid_;
  std::shared_ptr<const DijkstraPlanner> planner_;
  SplineSettings settings_;
  SplineCommandSettings command_settings_;
  std::mutex mutex_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace pathing
