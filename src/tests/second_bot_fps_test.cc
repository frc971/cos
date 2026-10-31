// Mirrors second_bot_main; replay sources and FPS measurement stay in this test.
#include <filesystem>
#include <atomic>
#include <array>
#include <iostream>
#include <string>
#include <unordered_set>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "camera/get_earliest_timestamp.h"
#include "control_loop/rio_clock.h"
#include "control_loop/connect_to_rio.h"
#include "control_loop/context_handoff_node.h"
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
#include "streamer/ball_pathing_streamer_node.h"
#include "streamer/position_estimate_rio_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, gamepiece, true, "Enable the gamepiece processing loop");
ABSL_FLAG(std::string, replay_log_path, "",
          "Replay timestamped JPEGs from front/left/right/dev_orin_color subfolders");
ABSL_FLAG(double, benchmark_seconds, 0.0,
          "Measure FPS for this duration after a five-second warmup; 0 disables");
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
ABSL_FLAG(std::string, ball_pathing_model,                         // NOLINT
          "/root/models/ball_unet_rgb.engine",                       // NOLINT
          "TensorRT U-Net engine; empty disables ball pathing");  // NOLINT
ABSL_FLAG(std::string, ball_pathing_input_mode, "rgb",
          "U-Net input mode: rgb or grayscale");
ABSL_FLAG(int, ball_pathing_stream_port, 5805,
          "Annotated ball pathing MJPEG port; 0 disables the stream");
ABSL_FLAG(bool, ball_pathing_stream_rotate_180, false,
          "Rotate the annotated image and overlay 180 degrees for display");
ABSL_FLAG(double, ball_probability_threshold, 0.5,                 // NOLINT
          "U-Net foreground probability threshold");             // NOLINT
ABSL_FLAG(double, ball_cluster_radius_m, 0.4,                      // NOLINT
          "Robot-frame radius used to group visible balls");      // NOLINT
ABSL_FLAG(double, ball_grid_resolution_m, 0.20,                    // NOLINT
          "Projected ball-evidence grid resolution in meters");   // NOLINT
ABSL_FLAG(int, ball_mask_sample_stride, 4,                         // NOLINT
          "Pixel stride used while projecting the segmentation mask"); // NOLINT
ABSL_FLAG(double, ball_maximum_range_m, 8.0,                       // NOLINT
          "Maximum projected ball range in meters");              // NOLINT
ABSL_FLAG(double, ball_forward_velocity_gain, 0.75,  // NOLINT
          "Forward/backward velocity gain in inverse seconds");  // NOLINT
ABSL_FLAG(double, ball_lateral_velocity_gain, 1.5,  // NOLINT
          "Left/right velocity gain in inverse seconds");  // NOLINT
ABSL_FLAG(double, ball_minimum_speed_mps, 1.0,  // NOLINT
          "Minimum speed toward a ball cluster in meters per second");  // NOLINT
ABSL_FLAG(double, ball_maximum_speed_mps, 1.5,                     // NOLINT
          "Maximum magnitude of the robot-relative velocity goal"); // NOLINT
ABSL_FLAG(double, ball_hold_seconds, 0.5,                          // NOLINT
          "Time to hold the last goal after losing its cluster"); // NOLINT
ABSL_FLAG(double, ball_switch_evidence_ratio, 1.25,                // NOLINT
          "Evidence ratio required to switch tracked clusters");  // NOLINT

namespace {

auto AddCameraPipeline(
    const std::string& config_path, int stream_port,
    control_loop::ControlLoop& control_loop,
    control_loop::ThreadPool& thread_pool,
    localization::UnambiguousSolverNode& solver_node,
    const std::shared_ptr<streamer::PositionEstimateRioStreamerNode>&
        rio_sender_node,
    bool pva_detection, const std::string& log_path,
    bool use_for_localization) -> std::shared_ptr<control_loop::INode> {
  const camera::UVCCameraConfig config{config_path};
  const std::string jpeg_channel = "jpeg_buffer:" + config.name;

  std::shared_ptr<control_loop::INode> camera_node;
  const auto replay_path = absl::GetFlag(FLAGS_replay_log_path);
  if (replay_path.empty()) {
    auto uvc = std::make_shared<camera::UVCCameraNode>(jpeg_channel, config);
    uvc->Start();
    camera_node = uvc;
  } else {
    camera_node = std::make_shared<camera::UVCDiskCameraNode>(
        (std::filesystem::path(replay_path) / config.name).string(), jpeg_channel,
        camera::GetEarliestTimestamp(
            (std::filesystem::path(replay_path) / "front").string()) - 2.0);
  }
  control_loop.RegisterDependencyNode(camera_node);

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

  if (!use_for_localization) return camera_node;
  rio_sender_node->AddCameraTimestamp(jpeg_channel);

  const std::string decoded_channel = "hardware_decoded_image:" + config.name;
  const std::string detections_channel =
      "hardware_apriltag_detections:" + config.name;
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
  return camera_node;
}

}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  if (!absl::GetFlag(FLAGS_replay_log_path).empty()) {
    control_loop::StartNetworktablesAsHost();
    control_loop::RioClock::EnableSimulation();
  } else {
    control_loop::StartNetworktables(971);
  }

  control_loop::ControlLoop control_loop(1ms);
  control_loop.SetMaxContext(absl::GetFlag(FLAGS_max_context));
  control_loop::ThreadPool thread_pool;
  control_loop::ControlLoop gamepiece_loop(20ms);
  control_loop::ThreadPool gamepiece_thread_pool(1);

  const std::vector<std::string> localization_paths{
      "/root/constants/second_bot/front_camera.json",
      "/root/constants/second_bot/left_camera.json",
      "/root/constants/second_bot/right_camera.json"};
  const std::string color_config_path = "/root/constants/second_bot/color.json";

  const double benchmark_seconds = absl::GetFlag(FLAGS_benchmark_seconds);
  CHECK_GE(benchmark_seconds, 0.0);
  std::chrono::steady_clock::time_point benchmark_start;
  std::array<std::atomic_size_t, 3> localization_frames{};
  std::atomic_size_t gamepiece_frames{0};
  const auto measuring = [&] {
    if (benchmark_seconds == 0) return false;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - benchmark_start).count();
    return elapsed >= 5.0 && elapsed < 5.0 + benchmark_seconds;
  };
  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(true);
  solver_node->RegisterCallback([&](const control_loop::Context& context) {
    if (!measuring()) return;
    constexpr std::array channels{
        "jpeg_buffer:front", "jpeg_buffer:left", "jpeg_buffer:right"};
    for (size_t i = 0; i < channels.size(); ++i) {
      if (context->GetMessage<camera::JpegBuffer>(channels[i])) {
        ++localization_frames[i];
      }
    }
  });
  control_loop.RegisterNode(solver_node);

  auto rio_sender_node =
      std::make_shared<streamer::PositionEstimateRioStreamerNode>(
          "pose_with_variance", "/COS");
  control_loop.RegisterNode(rio_sender_node);

  int port = 5801;
  const bool pva_detection = absl::GetFlag(FLAGS_pva_detection);
  const std::string image_log_path =
      absl::GetFlag(FLAGS_log_images) ? control_loop::GetLogPath() : "";
  for (const auto& path : localization_paths) {
    AddCameraPipeline(path, port++, control_loop, thread_pool, *solver_node,
                      rio_sender_node, pva_detection, image_log_path, true);
  }
  // Keep color capture, raw streaming, and logging independent of localization
  // and available even when the ball model is disabled.
  auto color_camera = AddCameraPipeline(
      color_config_path, port, control_loop, thread_pool, *solver_node,
      rio_sender_node, pva_detection, image_log_path, false);

  const std::string input_mode_name = absl::GetFlag(FLAGS_ball_pathing_input_mode);
  CHECK(input_mode_name == "rgb" || input_mode_name == "grayscale")
      << "ball_pathing_input_mode must be rgb or grayscale";
  const auto input_mode = input_mode_name == "rgb"
                              ? gamepiece::UnetInputMode::kRgb
                              : gamepiece::UnetInputMode::kGrayscale;
  const auto decode_format = input_mode == gamepiece::UnetInputMode::kRgb
                                 ? NVJPEG_OUTPUT_RGBI : NVJPEG_OUTPUT_Y;

  const std::string ball_model = absl::GetFlag(FLAGS_ball_pathing_model);
  const bool gamepiece_enabled =
      absl::GetFlag(FLAGS_gamepiece) && !ball_model.empty();
  if (gamepiece_enabled) {
    gamepiece_loop.RegisterDependencyNode(
        std::make_shared<control_loop::ContextHandoffNode>(color_camera));
    constexpr std::string_view kColorJpeg = "jpeg_buffer:dev_orin_color";
    constexpr std::string_view kBallImage = "ball_pathing:decoded_color";
    constexpr std::string_view kBallGoal = "ball_pathing/velocity_goal";
    auto ball_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        kColorJpeg, kBallImage, decode_format, gamepiece_thread_pool);
    gamepiece_loop.RegisterNode(ball_decode_node);

    gamepiece::BallPathingConfig ball_config{
        .probability_threshold = static_cast<float>(
            absl::GetFlag(FLAGS_ball_probability_threshold)),
        .mask_sample_stride = absl::GetFlag(FLAGS_ball_mask_sample_stride),
        .maximum_range_m = absl::GetFlag(FLAGS_ball_maximum_range_m),
        .cluster_radius_m = absl::GetFlag(FLAGS_ball_cluster_radius_m),
        .grid_resolution_m = absl::GetFlag(FLAGS_ball_grid_resolution_m),
        .forward_velocity_gain_per_s =
            absl::GetFlag(FLAGS_ball_forward_velocity_gain),
        .lateral_velocity_gain_per_s =
            absl::GetFlag(FLAGS_ball_lateral_velocity_gain),
        .minimum_speed_mps = absl::GetFlag(FLAGS_ball_minimum_speed_mps),
        .maximum_speed_mps = absl::GetFlag(FLAGS_ball_maximum_speed_mps),
        .switch_evidence_ratio =
            absl::GetFlag(FLAGS_ball_switch_evidence_ratio),
        .hold_duration = std::chrono::duration<double>(
            absl::GetFlag(FLAGS_ball_hold_seconds)),
    };
    auto ball_pathing_node = std::make_shared<gamepiece::BallPathingNode>(
        kBallImage, kBallGoal, ball_model, color_config_path, input_mode,
        gamepiece_thread_pool, ball_config);
    gamepiece_loop.RegisterNode(ball_pathing_node);
    ball_pathing_node->RegisterCallback([&](const control_loop::Context& context) {
      if (measuring() && context->GetMessage<camera::DecodedJpegBuffer>(
                             "ball_pathing:decoded_color")) {
        ++gamepiece_frames;
      }
    });
    const int ball_stream_port = absl::GetFlag(FLAGS_ball_pathing_stream_port);
    if (ball_stream_port != 0) {
      auto ball_streamer = std::make_shared<streamer::BallPathingStreamerNode>(
          kColorJpeg, kBallGoal, color_config_path, ball_config.cluster_radius_m,
          "/stream", ball_stream_port,
          absl::GetFlag(FLAGS_ball_pathing_stream_rotate_180));
      gamepiece_loop.RegisterNode(ball_streamer);
    }
    auto ball_goal_streamer =
        std::make_shared<streamer::BallVelocityGoalStreamerNode>(kBallGoal,
                                                                  "/COS");
    gamepiece_loop.RegisterNode(ball_goal_streamer);
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
    if (gamepiece_enabled) gamepiece_loop.EnableLatencyLog();
  }

  auto publications = control_loop.GetLogPublications();
  std::unordered_set<std::string> channels;
  for (const auto& publication : publications) {
    channels.insert(publication.GetChannel());
  }
  for (const auto& publication : gamepiece_loop.GetLogPublications()) {
    if (channels.insert(publication.GetChannel()).second) {
      publications.push_back(publication);
    }
  }
  auto wpilog_writer = std::make_shared<logging::WPILogWriter>(
      (std::filesystem::path(control_loop::GetLogPath()) / "cos.wpilog")
          .string(),
      publications);
  control_loop.EnableWPILog(wpilog_writer);
  benchmark_start = std::chrono::steady_clock::now();
  if (!absl::GetFlag(FLAGS_replay_log_path).empty()) {
    control_loop::RioClock::Restart();
  }
  if (gamepiece_enabled) {
    gamepiece_loop.EnableWPILog(wpilog_writer);
    gamepiece_loop.Start();
  }
  control_loop.Start();

  if (benchmark_seconds > 0) {
    while (!stop::StopRequested() &&
           std::chrono::steady_clock::now() - benchmark_start <
               std::chrono::duration<double>(5.0 + benchmark_seconds)) {
      std::this_thread::sleep_for(10ms);
    }
  } else {
    stop::WaitUntilStop();
  }

  // Keep source notifications flowing until the blocking handoff has stopped.
  if (gamepiece_enabled) gamepiece_loop.Stop();
  gamepiece_thread_pool.Shutdown();
  control_loop.Stop();
  thread_pool.Shutdown();
  if (benchmark_seconds > 0) {
    std::cout << "benchmark gamepiece=" << gamepiece_enabled
              << " seconds=" << benchmark_seconds
              << " front_fps=" << localization_frames[0].load() / benchmark_seconds
              << " left_fps=" << localization_frames[1].load() / benchmark_seconds
              << " right_fps=" << localization_frames[2].load() / benchmark_seconds
              << " gamepiece_frames=" << gamepiece_frames.load()
              << " gamepiece_fps=" << gamepiece_frames.load() / benchmark_seconds
              << std::endl;
  }
}
