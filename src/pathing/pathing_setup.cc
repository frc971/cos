#include "pathing/pathing_setup.h"

#include <limits>
#include <stdexcept>
#include <string>

#include <networktables/NetworkTableInstance.h>

#include "control_loop/control_loop.h"
#include "localization/robot_pose_receiver_node.h"
#include "pathing/navgrid_source_node.h"
#include "streamer/navgrid_streamer_node.h"
#include "streamer/networktables_command_streamer_node.h"
#include "utils/json.h"

namespace pathing {
namespace {
using localization::RobotPoseReceiverNode;

constexpr char kSpline[] = "pathing:spline";
constexpr char kCommand[] = "pathing:command";
constexpr char kSplineOverlay[] = "pathing:spline_overlay";
auto ReadInt(const nlohmann::json& object, const char* key) -> int {
  const auto& value = object.at(key);
  if (!value.is_number_integer())
    throw std::invalid_argument(std::string(key) + " must be an integer");
  const auto number = value.get<double>();
  if (number < std::numeric_limits<int>::min() ||
      number > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string(key) + " is out of range");
  return static_cast<int>(number);
}

}  // namespace

auto LoadPathingConfig(const std::filesystem::path& path)
    -> PathingConfig {
  const auto json = utils::ReadJson(path.string());
  const auto& topics = json.at("topics");
  const auto& streamer = json.at("streamer");

  PathingConfig config{
      .period = std::chrono::milliseconds(ReadInt(json, "period_ms")),
      .navgrid_path = json.at("navgrid").get<std::string>(),
      .pose_topic = topics.at("pose").get<std::string>(),
      .goal_topic = topics.at("goal").get<std::string>(),
      .command_topic = topics.at("command").get<std::string>(),
      .stream_navgrid = streamer.at("enabled").get<bool>(),
      .streamer_port = ReadInt(streamer, "port"),
      .field_image_path = streamer.at("field_image").get<std::string>(),
      .spline = {},
      .command = {}};
  config.spline.robot_radius_meters = json.at("robot_radius_m").get<double>();
  config.spline.turn_penalty = json.at("turn_penalty").get<double>();
  config.command.command_timeout =
      std::chrono::milliseconds(ReadInt(json, "command_timeout_ms"));
  config.command.lookahead = units::meter_t{json.at("lookahead_m").get<double>()};
  config.command.motion.max_translation =
      units::meters_per_second_t{json.at("max_translation_mps").get<double>()};
  config.command.motion.max_rotation =
      units::radians_per_second_t{json.at("max_rotation_radps").get<double>()};
  return config;
}

auto ConfigurePathing(control_loop::ControlLoop& loop,
                      const PathingConfig& config)
    -> std::shared_ptr<streamer::NetworkTablesCommandStreamerNode> {
  const auto period = config.period;
  const auto& pose_topic = config.pose_topic;
  const auto& goal_topic = config.goal_topic;
  const auto& command_topic = config.command_topic;
  const int port = config.streamer_port;
  if (config.stream_navgrid && (port < 1 || port > 65535))
    throw std::invalid_argument(
        "Navgrid streamer port must be between 1 and 65535");
  if (period.count() <= 0 || period > config.command.command_timeout ||
      !config.command.IsValid())
    throw std::invalid_argument("Invalid NetworkTables pathing settings");
  if (pose_topic.empty() || goal_topic.empty() || command_topic.empty() ||
      pose_topic.front() != '/' || goal_topic.front() != '/' ||
      command_topic.front() != '/' || pose_topic == goal_topic ||
      command_topic == pose_topic || command_topic == goal_topic)
    throw std::invalid_argument(
        "Pathing NT topics must be absolute and distinct");

  const auto navgrid =
      streamer::NavgridMessage::Load(config.navgrid_path);
  const auto instance = nt::NetworkTableInstance::GetDefault();
  auto input = std::make_shared<RobotPoseReceiverNode>(
      instance, pose_topic, goal_topic, period);
  auto spline = std::make_shared<DijkstraSplineNode>(
      RobotPoseReceiverNode::kInput, NavgridSourceNode::kNavgrid, kCommand,
      config.spline, config.command);
  auto output = std::make_shared<streamer::NetworkTablesCommandStreamerNode>(
      kCommand, command_topic, period, instance);

  loop.SetMaxContext(1);
  loop.RegisterDependencyNode(std::make_shared<NavgridSourceNode>(navgrid));
  loop.RegisterDependencyNode(input);
  loop.RegisterNode(spline);
  loop.RegisterNode(output);
  if (config.stream_navgrid) {
    spline->EnableSplineDebug(kSpline, kSplineOverlay);
    auto navgrid_streamer = std::make_shared<streamer::NavgridStreamerNode>(
        "/navgrid", port, config.field_image_path);
    navgrid_streamer->AddPathingInputs(
        NavgridSourceNode::kNavgrid, kSpline, RobotPoseReceiverNode::kInput,
        RobotPoseReceiverNode::kPose, kCommand);
    navgrid_streamer->AddPolyline(kSplineOverlay);
    loop.RegisterNode(navgrid_streamer);
  }
  return output;
}

}  // namespace pathing
