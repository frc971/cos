#include <filesystem>
#include <string>

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
#include "localization/position_estimate_sender_node.h"
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/jpeg_buffer_log_node.h"
#include "networktables/NetworkTableInstance.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "streamer/position_estimate_rio_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, pva_detection, true,                              // NOLINT
          "Use PVA for AprilTag detection instead of CPU");       // NOLINT
ABSL_FLAG(uint, max_context, 1,                                   // NOLINT
          "Maximum number of concurrent control-loop contexts");  // NOLINT
ABSL_FLAG(bool, latency_log, false,                               // NOLINT
          "Log control-loop latency and loops per second");       // NOLINT
ABSL_FLAG(                                                        // NOLINT
    bool, log_images, true,                                       // NOLINT
    "Log timestamped JPEG frames to the run's log directory in "  // NOLINT
    "per-camera subfolders");                                     // NOLINT

namespace {

void AddCameraPipeline(
    const std::string& config_path, int stream_port,
    control_loop::ControlLoop& control_loop,
    control_loop::ThreadPool& thread_pool,
    localization::UnambiguousSolverNode& solver_node,
    const std::shared_ptr<streamer::PositionEstimateRioStreamerNode>&
        rio_sender_node,
    bool pva_detection, const std::string& log_path) {
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

  if (!log_path.empty()) {
    const auto camera_log_folder =
        std::filesystem::path(log_path) / config.name;
    std::filesystem::create_directories(camera_log_folder);
    auto jpeg_buffer_logger_node = std::make_shared<logging::JpegBufferLogNode>(
        jpeg_channel, camera_log_folder.string(), thread_pool);
    control_loop.RegisterNode(jpeg_buffer_logger_node);
  }

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

  solver_node.AddCamera(detections_channel, camera::Intrinsics{config_path},
                        camera::Extrinsics{config_path}, control_loop);
}

}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  control_loop::StartNetworktables(971);

  control_loop::ControlLoop control_loop(1ms);
  control_loop.SetMaxContext(absl::GetFlag(FLAGS_max_context));
  control_loop::ThreadPool thread_pool;

  const std::vector<std::string> paths{
      "/root/constants/second_bot/front_camera.json",
      "/root/constants/second_bot/left_camera.json",
      "/root/constants/second_bot/right_camera.json",
      "/root/constants/second_bot/color.json"};

  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(false);
  control_loop.RegisterNode(solver_node);

  auto rio_sender_node =
      std::make_shared<streamer::PositionEstimateRioStreamerNode>(
          "pose_with_variance", "/COS");
  control_loop.RegisterNode(rio_sender_node);

  int port = 5801;
  const bool pva_detection = absl::GetFlag(FLAGS_pva_detection);
  const std::string image_log_path =
      absl::GetFlag(FLAGS_log_images) ? control_loop::GetLogPath() : "";
  for (const auto& path : paths) {
    AddCameraPipeline(path, port++, control_loop, thread_pool, *solver_node,
                      rio_sender_node, pva_detection, image_log_path);
  }

  auto networktables_instance = nt::NetworkTableInstance::Create();
  networktables_instance.StartServer();
  auto variance_calculator_node =
      std::make_shared<localization::VarianceCalculatorNode>(
          "pose", "pose_with_variance");
  control_loop.RegisterNode(variance_calculator_node);
  auto position_estimate_sender_node =
      std::make_shared<localization::PositionEstimateSenderNode>(
          "pose_with_variance", "Orin/localization", networktables_instance);
  position_estimate_sender_node->SetLogEstimates(true);
  control_loop.RegisterNode(position_estimate_sender_node);

  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>("pose");
  control_loop.RegisterNode(simulation_position_sender_node);
  if (absl::GetFlag(FLAGS_latency_log)) {
    control_loop.EnableLatencyLog();
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
  networktables_instance.StopServer();
  nt::NetworkTableInstance::Destroy(networktables_instance);
}
