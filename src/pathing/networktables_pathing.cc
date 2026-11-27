#include "pathing/networktables_pathing.h"

#include <stdexcept>

#include "control_loop/control_loop.h"
#include "localization/robot_pose_node.h"
#include "pathing/navgrid_source_node.h"
#include "streamer/navgrid_streamer_node.h"
#include "streamer/networktables_command_streamer_node.h"

namespace pathing {
using nt::NetworkTableInstance;

namespace {
using localization::RobotPoseNode;
constexpr char kSpline[] = "pathing:spline";
constexpr char kCommand[] = "pathing:command";
constexpr char kSplineOverlay[] = "pathing:spline_overlay";

auto CommandSettings(const NetworkTablesPathingSettings& settings)
    -> SplineCommandSettings {
  return {.motion = settings.motion,
          .lookahead = settings.lookahead,
          .command_timeout = settings.command_timeout};
}

}  // namespace

void NetworkTablesPathingSettings::Validate() const {
  if (navgrid_streamer &&
      (navgrid_streamer_port < 1 || navgrid_streamer_port > 65535))
    throw std::invalid_argument(
        "Navgrid streamer port must be between 1 and 65535");
  if (period.count() <= 0 || period > command_timeout ||
      !CommandSettings(*this).IsValid())
    throw std::invalid_argument("Invalid NetworkTables pathing settings");
  if (pose_topic.empty() || goal_topic.empty() || command_topic.empty() ||
      pose_topic.front() != '/' || goal_topic.front() != '/' ||
      command_topic.front() != '/' || pose_topic == goal_topic ||
      command_topic == pose_topic || command_topic == goal_topic)
    throw std::invalid_argument(
        "Pathing NT topics must be absolute and distinct");
}

class NetworkTablesPathing::Impl {
 public:
  Impl(const streamer::NavgridMessage& navgrid,
       const NetworkTablesPathingSettings& settings,
       nt::NetworkTableInstance instance)
      : loop(settings.period) {
    settings.Validate();
    auto input = std::make_shared<RobotPoseNode>(
        instance, settings.pose_topic, settings.goal_topic, settings.period);
    auto spline_settings = settings.spline;
    spline_settings.robot_radius_meters = settings.robot_radius_meters;
    auto spline = std::make_shared<DijkstraSplineNode>(
        RobotPoseNode::kInput, NavgridSourceNode::kNavgrid, kCommand,
        spline_settings, CommandSettings(settings));
    if (settings.navgrid_streamer)
      spline->EnableSplineDebug(kSpline, kSplineOverlay);
    output = std::make_shared<streamer::NetworkTablesCommandStreamerNode>(
        kCommand, settings.command_topic, settings.period, instance, input);
    loop.SetMaxContext(1);
    loop.RegisterDependencyNode(std::make_shared<NavgridSourceNode>(navgrid));
    loop.RegisterDependencyNode(input);
    loop.RegisterNode(spline);
    loop.RegisterNode(output);
    if (settings.navgrid_streamer) {
      auto streamer_node = std::make_shared<streamer::NavgridStreamerNode>(
          "/navgrid", settings.navgrid_streamer_port,
          settings.field_image_path);
      streamer_node->AddPathingInputs(kSpline, RobotPoseNode::kInput,
                                      RobotPoseNode::kPose, kCommand);
      streamer_node->AddPolyline(kSplineOverlay);
      loop.RegisterNode(streamer_node);
    }
  }
  control_loop::ControlLoop loop;
  std::shared_ptr<streamer::NetworkTablesCommandStreamerNode> output;
  bool started = false;
};

NetworkTablesPathing::NetworkTablesPathing(
    const streamer::NavgridMessage& navgrid,
    NetworkTablesPathingSettings settings, NetworkTableInstance instance)
    : impl_(std::make_unique<Impl>(navgrid, settings, instance)) {}

NetworkTablesPathing::~NetworkTablesPathing() {
  Stop();
}

void NetworkTablesPathing::Start() {
  if (impl_->started)
    return;
  impl_->loop.Start();
  impl_->started = true;
}

void NetworkTablesPathing::Stop() {
  if (!impl_->started)
    return;
  impl_->loop.Stop();
  impl_->started = false;
  impl_->output->PublishStop();
}
}  // namespace pathing
