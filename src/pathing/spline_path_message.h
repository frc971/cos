#pragma once

#include <cstddef>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <frc/spline/CubicHermiteSpline.h>

#include "control_loop/message.h"

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

}  // namespace pathing
