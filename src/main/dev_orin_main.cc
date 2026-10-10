#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loop/connect_to_rio.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/wpilog_writer.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "streamer/position_estimate_rio_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, pva_detection, true,                              // NOLINT
          "Use PVA for AprilTag detection instead of CPU");       // NOLINT
ABSL_FLAG(uint, max_context, 1,                                   // NOLINT
          "Maximum number of concurrent control-loop contexts");  // NOLINT

namespace {

void AddCameraPipeline(
    const std::string& config_path, int stream_port,
    control_loop::ControlLoop& control_loop,
    control_loop::ThreadPool& thread_pool,
    localization::UnambiguousSolverNode& solver_node,
    const std::shared_ptr<streamer::PositionEstimateRioStreamerNode>&
        rio_sender_node,
    bool pva_detection) {
  const camera::UVCCameraConfig config{config_path};
  const std::string jpeg_channel = "jpeg_buffer:" + config.name;
  const std::string decoded_channel = "hardware_decoded_image:" + config.name;
  const std::string detections_channel =
      "hardware_apriltag_detections:" + config.name;

  auto uvc_camera_node = std::make_shared<camera::UVCCameraNode>(
      jpeg_channel, camera::UVCCameraConfig{config_path});
  uvc_camera_node->Start();
  control_loop.RegisterDependencyNode(uvc_camera_node);
  rio_sender_node->AddCamera(*uvc_camera_node);

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
          decoded_channel, detections_channel, config_path, thread_pool,
          pva_detection);
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
  control_loop::StartNetworktables();

  control_loop::ControlLoop control_loop(1ms);
  control_loop.SetMaxContext(absl::GetFlag(FLAGS_max_context));
  control_loop::ThreadPool thread_pool;

  const std::vector<std::string> paths{"/root/constants/dev-orin/first.json",
                                       "/root/constants/dev-orin/second.json",
                                       "/root/constants/dev-orin/third.json"};

  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(false);
  control_loop.RegisterNode(solver_node);

  auto rio_sender_node =
      std::make_shared<streamer::PositionEstimateRioStreamerNode>("pose",
                                                                  "/COS");
  control_loop.RegisterNode(rio_sender_node);

  int port = 5801;
  const bool pva_detection = absl::GetFlag(FLAGS_pva_detection);
  for (const auto& path : paths) {
    AddCameraPipeline(path, port++, control_loop, thread_pool, *solver_node,
                      rio_sender_node, pva_detection);
  }

  auto variance_calculator_node =
      std::make_shared<localization::VarianceCalculatorNode>(
          "pose", "pose_with_variance");
  control_loop.RegisterNode(variance_calculator_node);
  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>(
          "pose_with_variance");
  control_loop.RegisterNode(simulation_position_sender_node);
  control_loop.EnableLatencyLog();

  auto wpilog_writer = std::make_shared<logging::WPILogWriter>(
      (std::filesystem::path(control_loop::GetLogPath()) / "cos.wpilog").string(),
      control_loop.GetLogPublications());
  control_loop.EnableWPILog(wpilog_writer);
  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
}
