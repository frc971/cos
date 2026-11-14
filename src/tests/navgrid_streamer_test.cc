#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "control_loop/control_loop.h"
#include "pathing/dijkstra_path_node.h"
#include "streamer/navgrid_streamer_node.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, navgrid,                                     // NOLINT
          "/root/constants/assets/navgrid.json",                    // NOLINT
          "Navgrid JSON file to stream.");                          // NOLINT
ABSL_FLAG(std::string, field_image,                                 // NOLINT
          "/root/constants/assets/field.png",                       // NOLINT
          "Field image to draw the navgrid on.");                   // NOLINT
ABSL_FLAG(int, port, 4971, "MJPEG server port.");                   // NOLINT
ABSL_FLAG(double, start_x, 1.0, "Starting field x in meters.");     // NOLINT
ABSL_FLAG(double, start_y, 1.0, "Starting field y in meters.");     // NOLINT
ABSL_FLAG(double, start_heading, 0.0,                               // NOLINT
          "Starting heading in degrees.");                          // NOLINT
ABSL_FLAG(double, goal_x, 15.0, "Goal field x in meters.");         // NOLINT
ABSL_FLAG(double, goal_y, 7.0, "Goal field y in meters.");          // NOLINT
ABSL_FLAG(double, goal_heading, 90.0, "Goal heading in degrees.");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  try {
    auto navgrid = std::make_shared<streamer::NavgridMessage>(
        absl::GetFlag(FLAGS_navgrid));
    pathing::DijkstraPathNode planner("pose", "goal", "velocity",
                                      absl::GetFlag(FLAGS_navgrid));
    planner.EnableNavgridDebug("route");
    streamer::NavgridStreamerNode streamer("/navgrid",
                                           absl::GetFlag(FLAGS_port),
                                           absl::GetFlag(FLAGS_field_image));
    streamer.AddNavgrid("navgrid");
    streamer.AddNavgrid("route");
    auto plan = planner.CreateCallback();
    auto stream = streamer.CreateCallback();
    frc::Pose2d pose{
        units::meter_t{absl::GetFlag(FLAGS_start_x)},
        units::meter_t{absl::GetFlag(FLAGS_start_y)},
        frc::Rotation2d{units::degree_t{absl::GetFlag(FLAGS_start_heading)}}};
    auto goal = std::make_shared<pathing::PoseMessage>(frc::Pose2d{
        units::meter_t{absl::GetFlag(FLAGS_goal_x)},
        units::meter_t{absl::GetFlag(FLAGS_goal_y)},
        frc::Rotation2d{units::degree_t{absl::GetFlag(FLAGS_goal_heading)}}});
    std::optional<pathing::PathStatus> last_status;
    constexpr auto period = std::chrono::milliseconds(100);
    const units::second_t step{std::chrono::duration<double>(period).count()};
    control_loop::ControlLoop loop(period);
    loop.RegisterCallback([&](const control_loop::Context& context) -> void {
      context->SetMessage("navgrid", navgrid);
      context->SetMessage("pose", std::make_shared<pathing::PoseMessage>(pose));
      context->SetMessage("goal", goal);
      plan(context);
      stream(context);
      const auto* velocity =
          context->GetMessage<pathing::VelocityGoalMessage>("velocity");
      if (last_status != velocity->status) {
        switch (velocity->status) {
          case pathing::PathStatus::FollowingPath:
            LOG(INFO) << "Dijkstra: following path (green), obstacles are red";
            break;
          case pathing::PathStatus::ReachedGoal:
            LOG(INFO) << "Dijkstra: reached goal";
            break;
          case pathing::PathStatus::InvalidInput:
            LOG(ERROR)
                << "Dijkstra: invalid start/goal; check field bounds and "
                   "blocked cells";
            break;
          case pathing::PathStatus::NoPath:
            LOG(ERROR) << "Dijkstra: no path to goal";
            break;
        }
        last_status = velocity->status;
      }
      // Simple field-relative simulation: apply this tick's velocity for 100 ms.
      pose = frc::Pose2d{
          pose.X() + velocity->vx * step, pose.Y() + velocity->vy * step,
          pose.Rotation() + frc::Rotation2d{velocity->omega * step}};
    });
    loop.Start();
    stop::WaitUntilStop();
    loop.Stop();
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return 1;
  }
  return 0;
}
