#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/camera_config.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/unambiguous_solver_node.h"

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// A latest-frame replay source with the same publication interface as a camera.
class ReplayCamera final : public control_loop::INode {
 public:
  ReplayCamera(std::string channel, std::vector<std::vector<uchar>> frames)
      : channel_(std::move(channel)), frames_(std::move(frames)),
        publications_{control_loop::MessageDescriptor::Publication<camera::JpegBuffer>(channel_)} {}
  auto CreateCallback() -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      const auto tick = static_cast<size_t>(control_loop::RioClock::GetTime() * 120);
      std::shared_ptr<camera::JpegBuffer> jpeg;
      if (tick != last_tick_) {
        last_tick_ = tick;
        const auto& bytes = frames_[tick % frames_.size()];
        jpeg = std::make_shared<camera::JpegBuffer>(bytes.size(), tick / 120.0);
        std::memcpy(jpeg->ptr, bytes.data(), bytes.size());
      }
      context->SetMessage(channel_, jpeg);
      for (const auto& callback : callbacks_) callback(context);
    };
  }
  void RegisterCallback(const std::function<void(const control_loop::Context&)>& callback) override { callbacks_.push_back(callback); }
  auto GetDependencies() const -> const std::vector<control_loop::MessageDescriptor>& override { return dependencies_; }
  auto GetPublications() const -> const std::vector<control_loop::MessageDescriptor>& override { return publications_; }
 private:
  std::string channel_;
  std::vector<std::vector<uchar>> frames_;
  size_t last_tick_ = 0;
  std::vector<control_loop::MessageDescriptor> dependencies_, publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

// Usage: temp_gamepiece_fps_main MANIFEST SECONDS [REPLAY_ROOT [--grayscale-only]]
// JPEGs are cached before timing; each measured frame includes decode and tracking.
auto Run(int argc, char** argv) -> int {
  absl::InitializeLog();
  CHECK(argc >= 3 && argc <= 5);
  const bool grayscale_only = argc == 5 && std::string(argv[4]) == "--grayscale-only";
  CHECK(argc != 5 || grayscale_only);
  const fs::path manifest_path = argv[1];
  nlohmann::json manifest;
  std::ifstream(manifest_path) >> manifest;
  const auto config_path = fs::temp_directory_path() / "gamepiece-fps-camera.json";
  std::ofstream(config_path) << manifest.at("camera_calibration");
  const camera::Intrinsics intrinsics(config_path);
  const camera::Extrinsics extrinsics(config_path);
  fs::remove(config_path);
  std::vector<std::vector<uchar>> frames;
  std::vector<frc::Pose3d> poses;
  for (const auto& frame : manifest.at("frames")) {
    const auto image = cv::imread((manifest_path.parent_path() /
                                  frame.at("file").get<std::string>()).string());
    CHECK(!image.empty());
    frames.emplace_back();
    CHECK(cv::imencode(".jpg", image, frames.back(), {cv::IMWRITE_JPEG_QUALITY, 95}));
    const auto& pose = frame.at("robot_pose");
    poses.emplace_back(units::meter_t{pose.at("translation_m")[0].get<double>()},
                       units::meter_t{pose.at("translation_m")[1].get<double>()},
                       units::meter_t{0}, frc::Rotation3d{units::radian_t{0},
                       units::radian_t{0}, units::radian_t{pose.at("rotation_rad").get<double>()}});
  }
  control_loop::ThreadPool pool(6);
  control_loop::ControlLoop localization_loop(1ms);
  auto solver = std::make_shared<localization::UnambiguousSolverNode>("pose");
  std::array<std::atomic<size_t>, 3> localization_frames{};
  std::atomic<size_t> pose_count{};
  solver->RegisterCallback([&](const control_loop::Context& context) {
    if (context->GetMessage<localization::PositionEstimateMessage>("pose")) ++pose_count;
  });
  control_loop::RioClock::EnableSimulation();
  if (argc >= 4) {
    for (size_t i = 0; i < 3; ++i) {
      const std::string name = std::array{"front", "left", "right"}[i];
      const auto calibration = "/root/constants/second_bot/" + name + "_camera.json";
      const auto jpeg = name + "/jpeg", decoded = name + "/decoded", tags = name + "/tags";
      // Separate grayscale JPEG camera streams, paced at 120 Hz. Cache input
      // before timing so this measures processing rather than disk bandwidth.
      std::vector<fs::path> paths;
      for (const auto& entry : fs::directory_iterator(fs::path(argv[3]) / name))
        if (entry.path().extension() == ".jpg" && !entry.path().filename().string().starts_with("."))
          paths.push_back(entry.path());
      std::sort(paths.begin(), paths.end(), [](const auto& a, const auto& b) {
        return std::stod(a.stem()) < std::stod(b.stem());
      });
      CHECK(!paths.empty());
      std::vector<std::vector<uchar>> gray_frames;
      for (size_t j = 0; j < std::min<size_t>(paths.size(), 120); ++j) {
        auto gray = cv::imread(paths[j].string(), cv::IMREAD_GRAYSCALE);
        CHECK(!gray.empty());
        CHECK(gray.cols == 1280 && gray.rows == 800);
        gray_frames.emplace_back();
        CHECK(cv::imencode(".jpg", gray, gray_frames.back(), {cv::IMWRITE_JPEG_QUALITY, 95}));
      }
      auto source = std::make_shared<ReplayCamera>(jpeg, std::move(gray_frames));
      auto decoder = std::make_shared<camera::NvjpegDecodeNode>(jpeg, decoded, NVJPEG_OUTPUT_Y, pool);
      auto detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(decoded, tags, calibration, pool);
      detector->RegisterCallback([&, i, tags](const control_loop::Context& context) {
        if (context->GetMessage<apriltag::TagDetections>(tags)) ++localization_frames[i];
      });
      localization_loop.RegisterDependencyNode(source);
      localization_loop.RegisterNode(decoder);
      localization_loop.RegisterNode(detector);
      solver->AddCameraTimestamp(jpeg);
      solver->AddCamera(tags, camera::Intrinsics(calibration), camera::Extrinsics(calibration), localization_loop);
    }
    localization_loop.RegisterNode(solver);
    localization_loop.Start();
  }
  if (grayscale_only) {
    std::this_thread::sleep_for(5s);
    std::array<size_t, 3> before;
    for (size_t i = 0; i < 3; ++i) before[i] = localization_frames[i];
    const auto poses_before = pose_count.load();
    const auto start = Clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(std::stod(argv[2])));
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    nlohmann::json rates = nlohmann::json::array(), counts = nlohmann::json::array();
    for (size_t i = 0; i < 3; ++i) {
      const size_t count = localization_frames[i] - before[i];
      CHECK_GT(count, 0U);
      counts.push_back(count);
      rates.push_back(count / seconds);
    }
    localization_loop.Stop();
    pool.Shutdown();
    std::cout << nlohmann::json{{"seconds", seconds}, {"localization_frames", counts},
        {"localization_fps", rates}, {"localization_poses", pose_count.load() - poses_before},
        {"mode", "grayscale_only"}}.dump() << std::endl;
    return 0;
  }
  camera::NvjpegDecodeNode decoder("jpeg", "decoded", NVJPEG_OUTPUT_BGRI, pool);
  const auto resolution = manifest.at("resolution");
  gamepiece::HSVClusterTrackerNode tracker("decoded", "clusters", intrinsics.ToMatrix(),
      extrinsics.ToCameraToRobot<cv::Matx44d>().inv(),
      resolution[0].get<size_t>() * resolution[1].get<size_t>(), "input_pose");
  decoder.RegisterCallback(tracker.CreateCallback());
  std::promise<size_t> completed;
  tracker.RegisterCallback([&](const control_loop::Context& context) {
    const auto* result = context->GetMessage<control_loop::ValueMessage<gamepiece::hsv_clusters_t>>("clusters");
    CHECK(result != nullptr);
    completed.set_value(tracker.GetClusters()->size());
  });
  auto decode = decoder.CreateCallback();
  std::vector<double> latencies;
  size_t index = 0, clusters = 0;
  auto run_frame = [&] {
    const auto start = Clock::now();
    auto context = std::make_shared<control_loop::ContextInternal>(start, nullptr, std::stop_token{}, index + 1);
    const auto i = index++ % frames.size();
    auto jpeg = std::make_shared<camera::JpegBuffer>(frames[i].size(), index / 100.0);
    std::memcpy(jpeg->ptr, frames[i].data(), frames[i].size());
    context->SetMessage("jpeg", jpeg);
    auto pose = std::make_shared<localization::PositionEstimateMessage>();
    pose->pose = poses[i];
    context->SetMessage("input_pose", pose);
    completed = std::promise<size_t>{};
    auto future = completed.get_future();
    decode(context);
    CHECK(future.wait_for(30s) == std::future_status::ready);
    clusters += future.get();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  const auto warmup = Clock::now();
  while (Clock::now() - warmup < 5s) run_frame();
  clusters = 0;
  std::array<size_t, 3> before;
  for (size_t i = 0; i < 3; ++i) before[i] = localization_frames[i];
  const auto poses_before = pose_count.load();
  const auto start = Clock::now();
  while (std::chrono::duration<double>(Clock::now() - start).count() < std::stod(argv[2])) latencies.push_back(run_frame());
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  nlohmann::json rates = nlohmann::json::array();
  for (size_t i = 0; i < 3; ++i) rates.push_back((localization_frames[i] - before[i]) / seconds);
  if (argc >= 4) for (size_t i = 0; i < 3; ++i) CHECK_GT(localization_frames[i].load(), before[i]);
  localization_loop.Stop();
  pool.Shutdown();
  std::sort(latencies.begin(), latencies.end());
  std::cout << nlohmann::json{{"frames", latencies.size()}, {"seconds", seconds},
      {"fps", latencies.size() / seconds}, {"mean_ms", seconds * 1000 / latencies.size()},
      {"p50_ms", latencies[latencies.size() / 2]}, {"p95_ms", latencies[latencies.size() * 95 / 100]},
      {"mean_clusters", double(clusters) / latencies.size()}, {"localization_fps", rates},
      {"localization_poses", pose_count.load() - poses_before}, {"resolution", resolution}}.dump() << std::endl;
  return 0;
}

auto main(int argc, char** argv) -> int {
  const int result = Run(argc, argv);
  // Avoid conflicting VPI/CUDA process finalizers after node destruction.
  std::_Exit(result);
}
