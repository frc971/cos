#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include <networktables/NetworkTableInstance.h>

#include "pathing/dijkstra_spline_node.h"

namespace pathing {

// These numeric values are part of the NetworkTables command wire format.
enum class PathingStatus : std::uint8_t {
  Inactive = 0,
  Following = 1,
  ReachedGoal = 2,
  InvalidInput = 3,
  StaleInput = 4,
  NoPath = 5,
  FitFailed = 6,
  Disconnected = 7,
};

struct NetworkTablesPathingSettings {
  double robot_radius_meters = 0.19;
  std::filesystem::path navgrid_path = "/root/constants/assets/navgrid.json";
  std::string pose_topic = "/DriveState/Pose";
  std::string goal_topic = "/COS/Pathing/Goal";
  std::string command_topic = "/COS/Pathing/Command";
  std::chrono::milliseconds period{20};
  std::chrono::milliseconds input_timeout{250};
  // The rio must stop holding a command within this interval. Translation is
  // capped so it cannot pass the collision-checked lookahead point in that time.
  std::chrono::milliseconds command_timeout{250};
  bool navgrid_streamer = true;
  int navgrid_streamer_port = 4975;
  std::filesystem::path field_image_path = "/root/constants/assets/field.png";
  units::meter_t lookahead{0.5};
  MotionSettings motion;
  SplineSettings spline;
};

// Independent NT input -> spline planner -> follower -> NT output loop.
// Uses an already connected instance; never creates a client/server itself.
class NetworkTablesPathing {
 public:
  // NT local time in microseconds; also called from NT listener threads.
  using Clock = std::function<int64_t()>;
  explicit NetworkTablesPathing(NetworkTablesPathingSettings settings = {},
                                nt::NetworkTableInstance instance =
                                    nt::NetworkTableInstance::GetDefault(),
                                Clock clock = nt::Now);
  ~NetworkTablesPathing();
  NetworkTablesPathing(const NetworkTablesPathing&) = delete;
  auto operator=(const NetworkTablesPathing&) -> NetworkTablesPathing& = delete;

  void Start();
  // Joins the worker before publishing an explicit zero/Inactive command.
  void Stop();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pathing
