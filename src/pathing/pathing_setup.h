#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include "pathing/dijkstra_spline_node.h"

namespace control_loop {
class ControlLoop;
}  // namespace control_loop

namespace streamer {
class NetworkTablesCommandStreamerNode;
}  // namespace streamer

namespace pathing {

struct PathingConfig {
  std::chrono::milliseconds period;
  std::filesystem::path navgrid_path;
  std::string pose_topic;
  std::string goal_topic;
  std::string command_topic;
  bool stream_navgrid;
  int streamer_port;
  std::filesystem::path field_image_path;
  SplineSettings spline;
  SplineCommandSettings command;
};

auto LoadPathingConfig(
    const std::filesystem::path& path = "/root/constants/vision-bot/pathing.json")
    -> PathingConfig;

// Register pathing nodes on a dedicated loop and return its command publisher.
auto ConfigurePathing(control_loop::ControlLoop& loop,
                      const PathingConfig& config)
    -> std::shared_ptr<streamer::NetworkTablesCommandStreamerNode>;

}  // namespace pathing
