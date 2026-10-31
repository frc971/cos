#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
#include "streamer/ball_pathing_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(std::string, log_path, "", "Directory of timestamped JPEG frames");
ABSL_FLAG(std::string, model_path, "/root/models/ball_unet_rgb.engine",
          "TensorRT U-Net engine; empty benchmarks RGB decode without U-Net");
ABSL_FLAG(bool, quiet, false, "Only print the replay summary");
ABSL_FLAG(std::string, ball_pathing_input_mode, "rgb",
          "U-Net input mode: rgb or grayscale");
ABSL_FLAG(std::string, camera_config,
          "/root/constants/second_bot/color.json",
          "Camera intrinsics and extrinsics JSON");
ABSL_FLAG(int, stream_port, 0, "Annotated MJPEG port; 0 disables streaming");
ABSL_FLAG(bool, stream_rotate_180, false,
          "Rotate annotated image and overlay 180 degrees for display");
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
  decode->EnableTiming("decode_latency");
  gamepiece::BallPathingConfig pathing_config;
  pathing_config.cluster_radius_m = absl::GetFlag(FLAGS_cluster_radius_m);
  std::shared_ptr<gamepiece::BallPathingNode> pathing;
  if (!absl::GetFlag(FLAGS_model_path).empty()) {
    pathing = std::make_shared<gamepiece::BallPathingNode>(
        kImage, kGoal, absl::GetFlag(FLAGS_model_path),
        absl::GetFlag(FLAGS_camera_config), input_mode, thread_pool, pathing_config);
  }

  std::atomic_size_t frames{0};
  std::atomic_size_t valid{0};
  std::vector<double> latency_ms, decode_ms, pathing_ms;
  std::chrono::steady_clock::time_point first, last;
  auto completed = [&](const control_loop::Context& context) {
    const auto* image = context->GetMessage<camera::DecodedJpegBuffer>(kImage);
    if (image == nullptr) return;
    const auto now = std::chrono::steady_clock::now();
    if (frames.load() == 0) first = now;
    last = now;
    latency_ms.push_back(
        std::chrono::duration<double, std::milli>(now - context->start).count());
    const double decode_time =
        context->GetMessage<control_loop::LatencyMessage>("decode_latency")
            ->latency.count() * 1000;
    decode_ms.push_back(decode_time);
    if (pathing) pathing_ms.push_back(latency_ms.back() - decode_time);
    ++frames;
    const auto* goal =
        context->GetMessage<gamepiece::BallVelocityGoalMessage>(kGoal);
    if (goal == nullptr) return;
    if (goal->valid) ++valid;
    if (absl::GetFlag(FLAGS_quiet)) return;
    std::cout << std::fixed << std::setprecision(6)
              << "goal timestamp=" << goal->timestamp
              << " valid=" << goal->valid << " held=" << goal->held
              << " target_m=(" << goal->target_x_m << ',' << goal->target_y_m
              << ") velocity_mps=(" << goal->vx_mps << ',' << goal->vy_mps
              << ") evidence_m2=" << goal->ball_evidence_m2
              << " balls=" << goal->estimated_ball_count << '\n';
  };
  if (pathing) pathing->RegisterCallback(completed);
  else decode->RegisterCallback(completed);

  loop.RegisterDependencyNode(disk);
  loop.RegisterNode(decode);
  if (pathing) loop.RegisterNode(pathing);
  if (pathing && absl::GetFlag(FLAGS_stream_port) != 0) {
    auto stream = std::make_shared<streamer::BallPathingStreamerNode>(
        kJpeg, kGoal, absl::GetFlag(FLAGS_camera_config),
        pathing_config.cluster_radius_m, "/stream", absl::GetFlag(FLAGS_stream_port),
        absl::GetFlag(FLAGS_stream_rotate_180));
    loop.RegisterNode(stream);
  }
  loop.Start();
  stop::WaitUntilStop();
  loop.Stop();
  thread_pool.Shutdown();
  const double elapsed = std::chrono::duration<double>(last - first).count();
  const auto percentile = [](std::vector<double>& values, double fraction) {
    std::sort(values.begin(), values.end());
    return values.empty() ? 0.0
        : values[static_cast<size_t>(fraction * (values.size() - 1))];
  };
  std::cout << std::fixed << std::setprecision(3)
            << "summary unet=" << static_cast<bool>(pathing)
            << " seconds=" << elapsed
            << " fps=" << (elapsed > 0 ? (frames.load() - 1) / elapsed : 0)
            << " latency_p50_ms=" << percentile(latency_ms, 0.5)
            << " latency_p95_ms=" << percentile(latency_ms, 0.95)
            << " decode_p50_ms=" << percentile(decode_ms, 0.5)
            << " pathing_p50_ms=" << percentile(pathing_ms, 0.5)
            << " pathing_p95_ms=" << percentile(pathing_ms, 0.95)
            << " frames=" << frames.load() << " valid=" << valid.load()
            << '\n';
  return (pathing ? valid.load() : frames.load()) > 0 ? 0 : 1;
}
