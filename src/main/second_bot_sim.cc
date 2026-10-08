#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/uvc_camera_config.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/wpilog_writer.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "control_loop/connect_to_rio.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(                                                // NOLINT
    std::string, log_path, "/cos-logs/second_bot/log16",  // NOLINT
    "Directory containing the left and right camera log directories");  // NOLINT
ABSL_FLAG(bool, reject_far_tags, true,                             // NOLINT
          "Reject AprilTags that are too small or too far away");  // NOLINT

namespace {

void AddCameraPipeline(const std::string& config_path,
                       const std::string& log_path, double replay_offset,
                       int stream_port, control_loop::ControlLoop& control_loop,
                       control_loop::ThreadPool& thread_pool,
                       localization::UnambiguousSolverNode& solver_node) {
  const camera::UVCCameraConfig config{config_path};
  const std::string jpeg_channel = "jpeg_buffer:" + config.name;
  const std::string decoded_channel = "hardware_decoded_image:" + config.name;
  const std::string detections_channel =
      "hardware_apriltag_detections:" + config.name;

  auto uvc_camera_node = std::make_shared<camera::UVCDiskCameraNode>(
      log_path, jpeg_channel, replay_offset);
  control_loop.RegisterDependencyNode(uvc_camera_node);

  auto jpeg_buffer_streamer_node =
      std::make_shared<streamer::JpegBufferStreamerNode>(
          jpeg_channel, "/stream", stream_port);
  control_loop.RegisterNode(jpeg_buffer_streamer_node);

  auto hardware_decode_node = std::make_shared<camera::NvjpegFdDecodeNode>(
      jpeg_channel, decoded_channel, thread_pool);
  control_loop.RegisterNode(hardware_decode_node);
  hardware_decode_node->EnableTiming("hardware_decoded_image:latency:" +
                                     config.name);

  auto hardware_apriltag_detector_node =
      std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
          decoded_channel, detections_channel, config_path, thread_pool);
  control_loop.RegisterNode(hardware_apriltag_detector_node);
  hardware_apriltag_detector_node->EnableTiming(
      "hardware_apriltag_detections:latency:" + config.name);

  solver_node.AddCameraTimestamp(jpeg_channel);
  solver_node.AddCamera(detections_channel, camera::Intrinsics{config_path},
                        camera::Extrinsics{config_path}, control_loop);
}

}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();
  control_loop::StartNetworktablesAsHost();

  control_loop::ControlLoop control_loop(100ms);
  control_loop::ThreadPool thread_pool;

  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(absl::GetFlag(FLAGS_reject_far_tags));

  control_loop.RegisterNode(solver_node);

  const std::filesystem::path log_path = absl::GetFlag(FLAGS_log_path);
  const std::vector<std::string> camera_log_paths = {
      (log_path / "front").string(),
      (log_path / "left").string(),
      (log_path / "right").string(),
  };
  const double replay_offset = camera::GetEarliestTimestamp(camera_log_paths);

  LOG(INFO) << replay_offset;
  AddCameraPipeline("/root/constants/second_bot/front_camera.json",
                    camera_log_paths[0], replay_offset, 5801, control_loop,
                    thread_pool, *solver_node);
  AddCameraPipeline("/root/constants/second_bot/left_camera.json",
                    camera_log_paths[1], replay_offset, 5802, control_loop,
                    thread_pool, *solver_node);
  AddCameraPipeline("/root/constants/second_bot/right_camera.json",
                    camera_log_paths[2], replay_offset, 5803, control_loop,
                    thread_pool, *solver_node);

  auto variance_calculator_node =
      std::make_shared<localization::VarianceCalculatorNode>(
          "pose", "pose_with_variance");
  control_loop.RegisterNode(variance_calculator_node);
  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>(
          "pose_with_variance");
  control_loop.RegisterNode(simulation_position_sender_node);

  auto wpilog_writer = std::make_shared<logging::WPILogWriter>(
      (std::filesystem::path(control_loop::GetLogPath()) / "cos.wpilog").string(),
      control_loop.GetLogPublications());
  control_loop.EnableWPILog(wpilog_writer);
  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
}
