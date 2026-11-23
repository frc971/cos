#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "camera/camera_config.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "gamepiece/hsv_cluster_tracker_node.h"
#include "streamer/cluster_debug_streamer_node.h"
#include "streamer/densest_cluster_rio_streamer_node.h"
#include "utils/cv_geometry.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, config, "/root/constants/second_bot/color.json", "Gamepiece camera configuration");
ABSL_FLAG(std::string, log_path, "", "Replay timestamped JPEG directory instead of a live camera");
ABSL_FLAG(std::string, nt_server, "10.9.71.2", "NT server supplying /Drive/Pose (read only)");
ABSL_FLAG(std::vector<std::string>, pose, {}, "Optional fixed field pose: x,y,yaw (meters, radians)");
ABSL_FLAG(int, port, 5806, "HTTP port: /stream ellipses, /hsv threshold mask, /clusters densest ellipse, /balls ball circles");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  try {
    const auto fixed = absl::GetFlag(FLAGS_pose);
    if (!fixed.empty() && fixed.size() != 3) throw std::invalid_argument("--pose requires x,y,yaw");
    // Preview publication stays in a local NT instance, never on the robot.
    const auto preview = nt::NetworkTableInstance::Create();
    auto pose_instance = nt::NetworkTableInstance::Create();
    nt::StructPublisher<frc::Pose2d> fixed_pose;
    if (fixed.empty()) {
      pose_instance.StartClient4("cluster_viewer");
      pose_instance.SetServer(absl::GetFlag(FLAGS_nt_server).c_str());
    } else {
      std::vector<double> values;
      for (const auto& value : fixed) {
        std::size_t end;
        const double parsed = std::stod(value, &end);
        if (end != value.size() || !std::isfinite(parsed)) throw std::invalid_argument("Invalid fixed pose");
        values.push_back(parsed);
      }
      fixed_pose = pose_instance.GetStructTopic<frc::Pose2d>("/Drive/Pose").Publish();
      fixed_pose.Set(frc::Pose2d{units::meter_t{values[0]}, units::meter_t{values[1]}, frc::Rotation2d{units::radian_t{values[2]}}});
    }
    const auto path = absl::GetFlag(FLAGS_config);
    const camera::UVCCameraConfig config{path};
    control_loop::ThreadPool workers(1);
    control_loop::ControlLoop loop(std::chrono::milliseconds(1));
    loop.SetMaxContext(1);
    auto decoder = std::make_shared<camera::NvjpegDecodeNode>("jpeg", "decoded", NVJPEG_OUTPUT_BGRI, workers);
    auto tracker = std::make_shared<gamepiece::HSVClusterTrackerNode>(
        "decoded", "clusters", camera::Intrinsics{path}.ToMatrix(),
        utils::EigenToCvMat(camera::Extrinsics{path}.ToRobotToCamera<frc::Transform3d>().ToMatrix()),
        static_cast<std::size_t>(config.width) * config.height, "",
        gamepiece::DensityHistory::kPredict, true, "/Drive/Pose", pose_instance);
    tracker->EnablePoseOutput("pose");
    tracker->EnableMaskOutput("mask");
    loop.RegisterNode(decoder);
    loop.RegisterNode(tracker);
    loop.RegisterNode(std::make_shared<streamer::DensestClusterRioStreamerNode>("clusters", "pose", "/COS/DensestCluster", preview));
    loop.RegisterNode(std::make_shared<streamer::ClusterDebugStreamerNode>("clusters", "pose", "/COS/DensestCluster", "mask", absl::GetFlag(FLAGS_port)));
    const auto log = absl::GetFlag(FLAGS_log_path);
    control_loop::RioClock::EnableSimulation();
    if (log.empty()) {
      auto camera = std::make_shared<camera::UVCCameraNode>("jpeg", config);
      camera->Start();
      loop.RegisterDependencyNode(camera);
    } else {
      loop.RegisterDependencyNode(std::make_shared<camera::UVCDiskCameraNode>(log, "jpeg", camera::GetEarliestTimestamp(log)));
    }
    LOG(INFO) << "Cluster viewer on :" << absl::GetFlag(FLAGS_port) << "/stream, /hsv, /clusters, /balls";
    loop.Start();
    stop::WaitUntilStop();
    loop.Stop();
    workers.Shutdown();
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return 1;
  }
  return 0;
}
