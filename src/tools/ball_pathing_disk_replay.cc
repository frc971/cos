#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/ball_pathing_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(std::string, log_path, "", "Directory of timestamped JPEG frames");
ABSL_FLAG(std::string, model_path, "/root/models/ball_unet.engine",
          "TensorRT U-Net engine");
ABSL_FLAG(std::string, ball_pathing_input_mode, "rgb",
          "U-Net input mode: rgb or grayscale");
ABSL_FLAG(std::string, camera_config,
          "/root/constants/second_bot/color.json",
          "Camera intrinsics and extrinsics JSON");
ABSL_FLAG(double, cluster_radius_m, 1.0,
          "Robot-frame radius used to score and centroid ball evidence");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  const std::string log_path = absl::GetFlag(FLAGS_log_path);
  CHECK(!log_path.empty());

  const std::string input_mode_name = absl::GetFlag(FLAGS_ball_pathing_input_mode);
  CHECK(input_mode_name == "rgb" || input_mode_name == "grayscale")
      << "ball_pathing_input_mode must be rgb or grayscale";
  const auto input_mode = input_mode_name == "rgb"
                              ? gamepiece::UnetInputMode::kRgb
                              : gamepiece::UnetInputMode::kGrayscale;
  const auto decode_format = input_mode == gamepiece::UnetInputMode::kRgb
                                 ? NVJPEG_OUTPUT_RGBI : NVJPEG_OUTPUT_Y;

  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();
  // Match the production vision loop so disk replay does not overwrite
  // closely spaced camera frames before they enter the decode pipeline.
  control_loop::ControlLoop loop(1ms);
  control_loop::ThreadPool thread_pool(2);

  constexpr std::string_view kJpeg = "ball_pathing/disk_jpeg";
  constexpr std::string_view kImage = "ball_pathing/disk_image";
  constexpr std::string_view kGoal = "ball_pathing/velocity_goal";
  auto disk = std::make_shared<camera::UVCDiskCameraNode>(
      log_path, kJpeg, camera::GetEarliestTimestamp(log_path));
  auto decode = std::make_shared<camera::NvjpegDecodeNode>(
      kJpeg, kImage, decode_format, thread_pool);
  gamepiece::BallPathingConfig pathing_config;
  pathing_config.cluster_radius_m = absl::GetFlag(FLAGS_cluster_radius_m);
  auto pathing = std::make_shared<gamepiece::BallPathingNode>(
      kImage, kGoal, absl::GetFlag(FLAGS_model_path),
      absl::GetFlag(FLAGS_camera_config), input_mode, thread_pool, pathing_config);

  std::atomic_size_t frames{0};
  std::atomic_size_t valid{0};
  pathing->RegisterCallback([&](const control_loop::Context& context) {
    const auto* goal =
        context->GetMessage<gamepiece::BallVelocityGoalMessage>(kGoal);
    if (goal == nullptr) return;
    ++frames;
    if (goal->valid) ++valid;
    std::cout << std::fixed << std::setprecision(6)
              << "goal timestamp=" << goal->timestamp
              << " valid=" << goal->valid << " held=" << goal->held
              << " target_m=(" << goal->target_x_m << ',' << goal->target_y_m
              << ") velocity_mps=(" << goal->vx_mps << ',' << goal->vy_mps
              << ") evidence_m2=" << goal->ball_evidence_m2
              << " balls=" << goal->estimated_ball_count << '\n';
  });

  loop.RegisterDependencyNode(disk);
  loop.RegisterNode(decode);
  loop.RegisterNode(pathing);
  loop.Start();
  stop::WaitUntilStop();
  loop.Stop();
  thread_pool.Shutdown();
  std::cout << "summary frames=" << frames.load() << " valid=" << valid.load()
            << '\n';
  return valid.load() > 0 ? 0 : 1;
}
