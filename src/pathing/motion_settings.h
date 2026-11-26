#pragma once

#include <units/angle.h>
#include <units/angular_velocity.h>
#include <units/length.h>
#include <units/velocity.h>

namespace pathing {

// Footprint, command limits, and proportional control settings for the follower.
struct MotionSettings {
  double robot_radius_meters = 0;
  units::meters_per_second_t max_translation{1};
  units::radians_per_second_t max_rotation{2};
  double translation_gain = 1;  // 1/s
  double heading_gain = 2;      // 1/s
  units::meter_t position_tolerance{0.05};
  units::radian_t heading_tolerance{0.05235987755982989};  // 3 degrees
  // Extra path cost in meters per radian of travel-direction change.
  // Zero disables the penalty. The first move has no incoming direction.
  double turn_penalty = 0.3;
};

}  // namespace pathing
