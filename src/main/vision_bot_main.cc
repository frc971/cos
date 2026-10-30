#include <filesystem>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loop/connect_to_rio.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/ball_pathing_node.h"
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/jpeg_buffer_log_node.h"
#include "logging/wpilog_writer.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "streamer/ball_velocity_goal_streamer_node.h"
#include "streamer/position_estimate_rio_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, pva_detection, true,                              // NOLINT
          "Use PVA for AprilTag detection instead of CPU");       // NOLINT
ABSL_FLAG(uint, max_context, 1,                                   // NOLINT
          "Maximum number of concurrent control-loop contexts");  // NOLINT
ABSL_FLAG(bool, latency_log, false,                               // NOLINT
          "Log control-loop latency and loops per second");       // NOLINT
ABSL_FLAG(  // NOLINT
    bool, log_images, false,                                      // NOLINT
    "Log timestamped JPEG frames to the run's log directory in "  // NOLINT
    "per-camera subfolders");                                     // NOLINT
ABSL_FLAG(std::string, ball_pathing_model,                         // NOLINT
          "/root/gamepiece_models/ball_unet.engine",             // NOLINT
          "FP32-I/O TensorRT engine for front-camera ball pathing; " // NOLINT
          "empty disables ball pathing");                         // NOLINT
ABSL_FLAG(double, ball_probability_threshold, 0.5,                 // NOLINT
          "U-Net foreground probability threshold");             // NOLINT
ABSL_FLAG(double, ball_cluster_radius_m, 1.0,                      // NOLINT
          "Robot-frame radius used to group visible balls");      // NOLINT
ABSL_FLAG(double, ball_grid_resolution_m, 0.20,                    // NOLINT
          "Projected ball-evidence grid resolution in meters");   // NOLINT
ABSL_FLAG(int, ball_mask_sample_stride, 4,                         // NOLINT
          "Pixel stride used while projecting the segmentation mask"); // NOLINT
ABSL_FLAG(double, ball_maximum_range_m, 8.0,                       // NOLINT
          "Maximum projected ball range in meters");             // NOLINT
ABSL_FLAG(double, ball_velocity_gain, 1.5,                         // NOLINT
          "Ball target proportional velocity gain in inverse seconds"); // NOLINT
ABSL_FLAG(double, ball_maximum_speed_mps, 1.5,                     // NOLINT
          "Maximum magnitude of the robot-relative velocity goal"); // NOLINT
ABSL_FLAG(double, ball_stop_radius_m, 0.30,                        // NOLINT
          "Target distance at which the velocity goal becomes zero"); // NOLINT
ABSL_FLAG(double, ball_hold_seconds, 0.5,                          // NOLINT
          "Time to hold the last goal after losing its cluster"); // NOLINT
ABSL_FLAG(double, ball_switch_evidence_ratio, 1.25,                // NOLINT
          "Evidence ratio required to switch tracked clusters");  // NOLINT

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
  control_loop::StartNetworktables(9972);

  control_loop::ControlLoop control_loop(1ms);
  control_loop.SetMaxContext(absl::GetFlag(FLAGS_max_context));
  control_loop::ThreadPool thread_pool;

  const std::vector<std::string> paths{"/root/constants/vision-bot/front.json",
                                       "/root/constants/vision-bot/left.json",
                                       "/root/constants/vision-bot/back.json",
                                       "/root/constants/vision-bot/right.json"};

  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(true);
  control_loop.RegisterNode(solver_node);

  auto rio_sender_node =
      std::make_shared<streamer::PositionEstimateRioStreamerNode>(
          "pose_with_variance", "/COS");
  control_loop.RegisterNode(rio_sender_node);

  int port = 4971;
  const bool pva_detection = absl::GetFlag(FLAGS_pva_detection);
  const std::string image_log_path =
      absl::GetFlag(FLAGS_log_images) ? control_loop::GetLogPath() : "";
  for (const auto& path : paths) {
    AddCameraPipeline(path, port++, control_loop, thread_pool, *solver_node,
                      rio_sender_node, pva_detection, image_log_path);
    LOG(INFO) << "Started camera: " << path << " with port " << port;
  }

  const std::string ball_model = absl::GetFlag(FLAGS_ball_pathing_model);
  if (!ball_model.empty()) {
    constexpr std::string_view kFrontJpeg =
        "jpeg_buffer:vision_bot_front";
    constexpr std::string_view kBallImage =
        "ball_pathing:grayscale_front";
    constexpr std::string_view kBallGoal =
        "ball_pathing/velocity_goal";
    auto ball_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        kFrontJpeg, kBallImage, NVJPEG_OUTPUT_Y, thread_pool);
    control_loop.RegisterNode(ball_decode_node);

    gamepiece::BallPathingConfig ball_config{
        .probability_threshold = static_cast<float>(
            absl::GetFlag(FLAGS_ball_probability_threshold)),
        .mask_sample_stride = absl::GetFlag(FLAGS_ball_mask_sample_stride),
        .maximum_range_m = absl::GetFlag(FLAGS_ball_maximum_range_m),
        .cluster_radius_m = absl::GetFlag(FLAGS_ball_cluster_radius_m),
        .grid_resolution_m = absl::GetFlag(FLAGS_ball_grid_resolution_m),
        .velocity_gain_per_s = absl::GetFlag(FLAGS_ball_velocity_gain),
        .maximum_speed_mps = absl::GetFlag(FLAGS_ball_maximum_speed_mps),
        .stop_radius_m = absl::GetFlag(FLAGS_ball_stop_radius_m),
        .switch_evidence_ratio =
            absl::GetFlag(FLAGS_ball_switch_evidence_ratio),
        .hold_duration = std::chrono::duration<double>(
            absl::GetFlag(FLAGS_ball_hold_seconds)),
    };
    auto ball_pathing_node = std::make_shared<gamepiece::BallPathingNode>(
        kBallImage, kBallGoal, ball_model,
        "/root/constants/vision-bot/front.json", thread_pool, ball_config);
    control_loop.RegisterNode(ball_pathing_node);
    auto ball_goal_streamer =
        std::make_shared<streamer::BallVelocityGoalStreamerNode>(kBallGoal,
                                                                  "/COS");
    control_loop.RegisterNode(ball_goal_streamer);
  }

  auto variance_calculator_node =
      std::make_shared<localization::VarianceCalculatorNode>(
          "pose", "pose_with_variance");
  control_loop.RegisterNode(variance_calculator_node);
  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>(
          "pose_with_variance");
  control_loop.RegisterNode(simulation_position_sender_node);
  if (absl::GetFlag(FLAGS_latency_log)) {
    control_loop.EnableLatencyLog();
  }

  auto wpilog_writer = std::make_shared<logging::WPILogWriter>(
      (std::filesystem::path(control_loop::GetLogPath()) / "cos.wpilog").string(),
      control_loop.GetLogPublications());
  control_loop.EnableWPILog(wpilog_writer);
  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
}
