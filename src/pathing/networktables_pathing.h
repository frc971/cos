#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include <networktables/NetworkTableInstance.h>

#include "pathing/dijkstra_spline_node.h"
#include "pathing/motion_settings.h"
#include "pathing/pathing_status.h"

namespace pathing {

// Topics, update cadence, planner settings, and optional MJPEG visualization.
struct NetworkTablesPathingSettings {
  // Footprint radius shared by the live planner and follower.
  double robot_radius_meters = 0.19;
  std::string pose_topic = "/DriveState/Pose";
  std::string goal_topic = "/COS/Pathing/Goal";
  std::string command_topic = "/COS/Pathing/Command";
  std::chrono::milliseconds period{20};
  // Retained for configuration compatibility. Inputs no longer expire.
  std::chrono::milliseconds input_timeout{250};
  // The rio must stop holding a command within this interval. Translation is
  // capped so it cannot pass the collision-checked lookahead point in that time.
  std::chrono::milliseconds command_timeout{250};
  bool navgrid_streamer = true;
  int navgrid_streamer_port = 4975;
  std::filesystem::path field_image_path = "/root/constants/assets/field.png";
  // Target distance ahead on the fitted path for the translation controller.
  units::meter_t lookahead{0.5};
  MotionSettings motion;
  SplineSettings spline;
};

// Independent NT input -> spline planner -> follower -> NT output loop.
// Uses an already connected instance; never creates a client/server itself.
class NetworkTablesPathing {
 public:
  // NT local time in microseconds.
  using Clock = std::function<int64_t()>;
  explicit NetworkTablesPathing(const streamer::NavgridMessage& navgrid,
                                NetworkTablesPathingSettings settings = {},
                                nt::NetworkTableInstance instance =
                                    nt::NetworkTableInstance::GetDefault(),
                                Clock clock = nt::Now);
  ~NetworkTablesPathing();
  NetworkTablesPathing(const NetworkTablesPathing&) = delete;
  auto operator=(const NetworkTablesPathing&) -> NetworkTablesPathing& = delete;

  // Starts periodic planning and command publication.
  void Start();
  // Joins the worker before publishing an explicit zero/Inactive command.
  void Stop();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pathing
