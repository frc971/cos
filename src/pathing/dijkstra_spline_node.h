#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <frc/spline/CubicHermiteSpline.h>

#include "control_loop/node.h"
#include "pathing/dijkstra_planner.h"
#include "pathing/pose_message.h"

namespace pathing {

enum class SplineStatus { Ready, InvalidInput, NoPath, FitFailed };

// Field-relative route geometry. Only Ready indicates a usable fit.
class SplinePathMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(SplinePathMessage);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) +
           (raw_route.size() + waypoints.size() + samples.size()) *
               sizeof(frc::Translation2d) +
           splines.size() * sizeof(frc::CubicHermiteSpline);
  }

  // Collision geometry for the exact navgrid snapshot used to plan this route.
  std::shared_ptr<const DijkstraPlanner> planner;
  SplineStatus status = SplineStatus::InvalidInput;
  // Grid route before visibility simplification. Retained if fitting fails.
  std::vector<frc::Translation2d> raw_route;
  // Interpolation knots, including any points inserted during refinement.
  std::vector<frc::Translation2d> waypoints;
  // Evaluate segment i at t in [0, 1]; rotations are travel tangents.
  // Joins are C2 in cumulative waypoint-chord length, not segment index.
  std::vector<frc::CubicHermiteSpline> splines;
  // Ordered points along the fitted curve for rendering.
  std::vector<frc::Translation2d> samples;
  size_t refinement_passes = 0;
};

// Footprint, search cost, and limits on spline refinement and sampling.
struct SplineSettings {
  double robot_radius_meters = 0;
  double turn_penalty = 0.3;  // meters/radian, as in MotionSettings
  size_t max_refinement_passes = 32;
  size_t max_control_points = 4096;
  units::meter_t sample_spacing{0.02};
};

// Consumes measured and goal PoseMessages and a NavgridMessage, then publishes
// collision-checked SplinePathMessage geometry without timing or velocity commands.
class DijkstraSplineNode final : public control_loop::INode {
 public:
  DijkstraSplineNode(std::string_view pose_channel,
                     std::string_view goal_channel,
                     std::string_view output_channel,
                     std::string_view navgrid_channel,
                     SplineSettings settings = {});
  // Publish curve samples and waypoint markers as a PolylineMessage.
  // Call before registering with the control loop.
  void EnableSplineDebug(std::string_view output_channel);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  auto Calculate(const PoseMessage* pose, const PoseMessage* goal,
                 const DijkstraPlanner& planner) const -> SplinePathMessage;

  std::string pose_channel_;
  std::string goal_channel_;
  std::string output_channel_;
  std::string navgrid_channel_;
  std::string debug_path_;
  std::shared_ptr<const streamer::NavgridMessage> navgrid_;
  std::shared_ptr<const DijkstraPlanner> planner_;
  SplineSettings settings_;
  std::mutex mutex_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace pathing
