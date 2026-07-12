// Full pipeline integration tests.
//
// Localization path:
//   UVC MJPEG frame -> camera ingestion -> LocalizationLoopController
//   -> IDecodeNode (NvjpegDecodeNode or OpenCVDecodeNode)
//   -> OpenCVApriltagDetectorNode -> UnambiguousSolverNode
//   -> SimSender / NetworkTableSender
//
// The decode node is injected via the IDecodeNode interface, so the same
// pipeline plumbing runs either against NVIDIA's hardware decoder or, fully
// on CPU, against OpenCVDecodeNode -- see the "OnCpu" tests below, which run
// unconditionally (no NVIDIA runtime required).
//
// Gamepiece path (requires CUDA GPU + model file):
//   UVC MJPEG frame -> camera ingestion -> LocalizationLoopController
//   -> NvjpegDecodeNode -> GamepieceLoopController -> YoloNode
//   Skips at runtime when prerequisites are absent.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "apriltag/opencv_apriltag_detector_node.h"
#include "camera/decoded_jpeg_buffer.h"
#include "camera/disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/opencv_decode_node.h"
#include "control_loops/loop_controller.h"
#include "gamepiece/yolo_node.h"
#include "gtest/gtest.h"
#include "localization/networktable_sender.h"
#include "localization/sim_sender.h"
#include "localization/unambiguous_solver_node.h"
#include "unit_tests/cuda_test_helpers.h"
#include "unit_tests/test_helpers.h"
#include "wpi/datalog/DataLogWriter.hpp"
#include "wpi/nt/NetworkTableInstance.hpp"

#include <link.h>

namespace {

constexpr char kEncodedJpegPath[] =
    COS_SOURCE_DIR "/unit_tests/testdata/jpeg_frames/encoded/20.283378.jpg";

// Place a TensorRT .engine file here to enable the gamepiece test.
constexpr char kGamepieceModelPath[] =
    COS_SOURCE_DIR "/unit_tests/testdata/gamepiece/model.engine";

auto LinkedJpegLibraryPath() -> std::string {
  std::string path;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        std::string* path = static_cast<std::string*>(data);
        const std::string name =
            info->dlpi_name == nullptr ? "" : info->dlpi_name;
        if (name.find("libjpeg.so") != std::string::npos) {
          *path = name;
          return 1;
        }
        return 0;
      },
      &path);
  return path;
}

// Recorded-log frame filenames encode a "seconds.microseconds" capture
// timestamp, e.g. "10.039759.jpg" -> 10039759us. Reconstruct the same
// microsecond integer used elsewhere in the pipeline (see kEncodedJpegPath
// above, whose stem "20.283378" is passed around as the literal 20283378).
auto ParseFilenameTimestampMicros(const std::string& stem) -> unsigned long {
  const size_t dot = stem.find('.');
  CHECK(dot != std::string::npos) << "Unexpected frame filename: " << stem;
  std::string frac = stem.substr(dot + 1);
  frac.resize(6, '0');
  return std::stoul(stem.substr(0, dot)) * 1'000'000UL + std::stoul(frac);
}

auto ListSortedFrames(const std::filesystem::path& dir)
    -> std::vector<std::pair<std::filesystem::path, unsigned long>> {
  std::vector<std::pair<std::filesystem::path, unsigned long>> frames;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() != ".jpg") {
      continue;
    }
    frames.emplace_back(
        entry.path(), ParseFilenameTimestampMicros(entry.path().stem().string()));
  }
  std::sort(frames.begin(), frames.end(),
           [](const auto& a, const auto& b) { return a.second < b.second; });
  return frames;
}

auto HasNvidiaJpegRuntime() -> bool {
  const std::string jpeg_path = LinkedJpegLibraryPath();
  return jpeg_path.find("/tegra/") != std::string::npos ||
         jpeg_path.find("nvidia") != std::string::npos;
}

// Wired localization pipeline with no solver callback pre-registered. Call
// solver->RegisterCallback() after construction, then emit camera frames.
struct LocalizationPipeline {
  std::unique_ptr<camera::ICamera> camera;
  std::shared_ptr<control_loops::LocalizationLoopController> controller;
  std::unique_ptr<camera::IDecodeNode> decoder;
  std::unique_ptr<apriltag::OpenCVApriltagDetectorNode> detector;
  std::unique_ptr<localization::UnambiguousSolverNode> solver;
};

auto MakeLocalizationPipeline(const std::filesystem::path& frame_path,
                              unsigned long timestamp,
                              const std::vector<camera::camera_constant_t>&
                                  cameras,
                              std::unique_ptr<camera::IDecodeNode> decoder)
    -> LocalizationPipeline {
  LocalizationPipeline p;
  p.camera = std::make_unique<camera::DiskCamera>(frame_path, timestamp);
  p.controller = std::make_shared<control_loops::LocalizationLoopController>(1);
  p.decoder = std::move(decoder);
  p.solver = std::make_unique<localization::UnambiguousSolverNode>(cameras);
  p.detector = std::make_unique<apriltag::OpenCVApriltagDetectorNode>(
      cos_test::testing::IntrinsicsJson());

  auto* detector_ptr = p.detector.get();
  p.decoder->RegisterCallback(
      [detector_ptr](std::shared_ptr<camera::DecodedJpegNvBuffer> frame,
                     control_loops::MetaDataList metadata,
                     std::shared_ptr<control_loops::Context> ctx) {
        detector_ptr->Detect(std::move(frame), std::move(metadata),
                             std::move(ctx));
      });

  auto* solver_ptr = p.solver.get();
  p.detector->RegisterCallback(
      [solver_ptr](
          std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context> ctx) {
        solver_ptr->Accumulate(std::move(detections), std::move(metadata),
                               std::move(ctx));
      });

  auto* decoder_ptr = p.decoder.get();
  p.controller->RegisterIterationCallback(
      0, [decoder_ptr](std::shared_ptr<camera::JpegBuffer> jpeg,
                       control_loops::MetaDataList metadata,
                       std::shared_ptr<control_loops::Context> ctx) {
        decoder_ptr->Decode(std::move(jpeg), std::move(metadata),
                            std::move(ctx));
      });

  p.camera->RegisterCallback(
      [controller = p.controller](std::shared_ptr<camera::JpegBuffer> frame,
                                  unsigned long timestamp) {
        controller->ReceiveFrame(0, std::move(frame), timestamp);
      });

  return p;
}

void RunUntilDone(LocalizationPipeline& pipeline, const std::atomic<bool>& done,
                  int timeout_ms = 5000) {
  pipeline.camera->Start();
  std::jthread run_thread([&pipeline] { pipeline.controller->Run(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  pipeline.controller->RequestStop();
}

// -----------------------------------------------------------------------
// Single-frame localization smoke test
// -----------------------------------------------------------------------

TEST(LocalizationPipelineTest, ProducesEstimateFromDiskFrames) {
  if (!HasNvidiaJpegRuntime()) {
    GTEST_SKIP() << "Full localization pipeline requires NVIDIA's libjpeg "
                 << "runtime; linked libjpeg is " << LinkedJpegLibraryPath();
  }

  const auto tmp = std::filesystem::temp_directory_path();
  const std::string intrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "pipeline_intrinsics.json", cos_test::testing::IntrinsicsJson());
  const std::string extrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "pipeline_extrinsics.json", cos_test::testing::ExtrinsicsJson());

  ASSERT_TRUE(std::filesystem::exists(kEncodedJpegPath))
      << "Missing MJPEG camera fixture: " << kEncodedJpegPath;

  const std::vector<camera::camera_constant_t> cameras = {
      cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                            extrinsics_path)};
  auto pipeline = MakeLocalizationPipeline(
      kEncodedJpegPath, 20283378, cameras,
      std::make_unique<camera::NvjpegDecodeNode>("pipeline_test"));

  localization::SimSender sender;
  std::atomic<bool> done{false};
  pipeline.solver->RegisterCallback(
      [&](localization::position_estimate_t estimate,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context>) {
        sender.Send(estimate, std::move(metadata), nullptr);
        done = true;
        pipeline.controller->RequestStop();
      });

  RunUntilDone(pipeline, done);

  ASSERT_TRUE(sender.last_estimate.has_value());
  EXPECT_GT(sender.last_estimate->num_tags, 0);
  EXPECT_FALSE(sender.last_estimate->invalid);
}

// Same smoke test as above, but decoding with OpenCVDecodeNode instead of
// NvjpegDecodeNode. Since OpenCVApriltagDetectorNode is also CPU-only, this
// exercises the entire localization pipeline without any NVIDIA hardware
// dependency, and so runs unconditionally.
TEST(LocalizationPipelineTest, ProducesEstimateFromDiskFramesOnCpu) {
  const auto tmp = std::filesystem::temp_directory_path();
  const std::string intrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "pipeline_cpu_intrinsics.json", cos_test::testing::IntrinsicsJson());
  const std::string extrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "pipeline_cpu_extrinsics.json", cos_test::testing::ExtrinsicsJson());

  ASSERT_TRUE(std::filesystem::exists(kEncodedJpegPath))
      << "Missing MJPEG camera fixture: " << kEncodedJpegPath;

  const std::vector<camera::camera_constant_t> cameras = {
      cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                            extrinsics_path)};
  auto pipeline = MakeLocalizationPipeline(
      kEncodedJpegPath, 20283378, cameras,
      std::make_unique<camera::OpenCVDecodeNode>("pipeline_test_cpu"));

  localization::SimSender sender;
  std::atomic<bool> done{false};
  pipeline.solver->RegisterCallback(
      [&](localization::position_estimate_t estimate,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context>) {
        sender.Send(estimate, std::move(metadata), nullptr);
        done = true;
        pipeline.controller->RequestStop();
      });

  RunUntilDone(pipeline, done);

  ASSERT_TRUE(sender.last_estimate.has_value());
  EXPECT_GT(sender.last_estimate->num_tags, 0);
  EXPECT_FALSE(sender.last_estimate->invalid);
}

// -----------------------------------------------------------------------
// Full localization integration test: publishes N estimates via
// NetworkTableSender and logs them to a .wpilog file, mirroring what the
// robot runtime does.
// -----------------------------------------------------------------------

TEST(LocalizationIntegrationTest, WritesWpilogViaNetworkTableSender) {
  if (!HasNvidiaJpegRuntime()) {
    GTEST_SKIP() << "Full localization pipeline requires NVIDIA's libjpeg "
                 << "runtime; linked libjpeg is " << LinkedJpegLibraryPath();
  }

  const auto wpilog_dir = std::filesystem::path(COS_BINARY_DIR) / "wpilogs";
  std::filesystem::create_directories(wpilog_dir);
  const std::string wpilog_path =
      (wpilog_dir / "WritesWpilogViaNetworkTableSender.wpilog").string();
  std::filesystem::remove(wpilog_path);

  const auto tmp = std::filesystem::temp_directory_path();
  const std::string intrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "integration_intrinsics.json", cos_test::testing::IntrinsicsJson());
  const std::string extrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "integration_extrinsics.json", cos_test::testing::ExtrinsicsJson());

  ASSERT_TRUE(std::filesystem::exists(kEncodedJpegPath))
      << "Missing MJPEG camera fixture: " << kEncodedJpegPath;

  constexpr int kTargetEstimates = 5;
  std::atomic<int> estimates_received{0};

  // Inner scope: DataLogWriter buffers writes in a raw_ostream that is only
  // flushed to disk when the object is destroyed.  Everything that touches the
  // DataLog or NT instance must be torn down before we exit this scope.
  {
    // Dedicated NT instance so this test is isolated from other NT activity.
    // DataLogWriter mirrors every NT entry update to a .wpilog file on disk.
    auto nt_instance = wpi::nt::NetworkTableInstance::Create();
    nt_instance.StartLocal();
    std::error_code ec;
    wpi::log::DataLogWriter data_log(wpilog_path,
                                     ec);  // opened second; destroyed first
    ASSERT_FALSE(ec) << "Failed to open DataLog at " << wpilog_path << ": "
                     << ec.message();
    NT_DataLogger nt_logger =
        nt_instance.StartEntryDataLog(data_log, "", "NT:");

    const std::vector<camera::camera_constant_t> cameras = {
        cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                              extrinsics_path)};
    auto pipeline = MakeLocalizationPipeline(
        kEncodedJpegPath, 20283378, cameras,
        std::make_unique<camera::NvjpegDecodeNode>("pipeline_test"));
    auto sender = std::make_unique<localization::NetworkTableSender>(
        "front", nt_instance);
    std::atomic<bool> done{false};

    pipeline.solver->RegisterCallback(
        [&](localization::position_estimate_t estimate,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context>) {
          sender->Send(estimate, std::move(metadata), nullptr);
          nt_instance.FlushLocal();
          if (estimates_received.fetch_add(1) + 1 >= kTargetEstimates) {
            done = true;
            pipeline.controller->RequestStop();
          }
        });

    RunUntilDone(pipeline, done);

    // Tear down in dependency order before the scope closes.
    sender.reset();
    wpi::nt::NetworkTableInstance::StopEntryDataLog(nt_logger);
    nt_instance.StopLocal();
    wpi::nt::NetworkTableInstance::Destroy(nt_instance);
    // data_log destroyed here (last declared → first destroyed in scope LIFO):
    // its raw_ostream destructor flushes all buffered bytes to disk.
  }

  EXPECT_GE(estimates_received.load(), kTargetEstimates)
      << "Timed out before reaching " << kTargetEstimates << " estimates";
  ASSERT_TRUE(std::filesystem::exists(wpilog_path))
      << "Expected wpilog at " << wpilog_path;
  EXPECT_GT(std::filesystem::file_size(wpilog_path), 1000u)
      << "wpilog at " << wpilog_path << " is suspiciously small";
}

// Same integration test as above, but decoding with OpenCVDecodeNode instead
// of NvjpegDecodeNode, so it exercises the full robot-runtime pipeline --
// LoopController, NetworkTableSender, DataLogWriter -- entirely on CPU and
// runs unconditionally (no NVIDIA runtime required).
TEST(LocalizationIntegrationTest, WritesWpilogViaNetworkTableSenderOnCpu) {
  const auto wpilog_dir = std::filesystem::path(COS_BINARY_DIR) / "wpilogs";
  std::filesystem::create_directories(wpilog_dir);
  const std::string wpilog_path =
      (wpilog_dir / "WritesWpilogViaNetworkTableSenderOnCpu.wpilog").string();
  std::filesystem::remove(wpilog_path);

  const auto tmp = std::filesystem::temp_directory_path();
  const std::string intrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "integration_cpu_intrinsics.json",
      cos_test::testing::IntrinsicsJson());
  const std::string extrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "integration_cpu_extrinsics.json",
      cos_test::testing::ExtrinsicsJson());

  ASSERT_TRUE(std::filesystem::exists(kEncodedJpegPath))
      << "Missing MJPEG camera fixture: " << kEncodedJpegPath;

  constexpr int kTargetEstimates = 5;
  std::atomic<int> estimates_received{0};

  // Inner scope: DataLogWriter buffers writes in a raw_ostream that is only
  // flushed to disk when the object is destroyed.  Everything that touches the
  // DataLog or NT instance must be torn down before we exit this scope.
  {
    // Dedicated NT instance so this test is isolated from other NT activity.
    // DataLogWriter mirrors every NT entry update to a .wpilog file on disk.
    auto nt_instance = wpi::nt::NetworkTableInstance::Create();
    nt_instance.StartLocal();
    std::error_code ec;
    wpi::log::DataLogWriter data_log(wpilog_path,
                                     ec);  // opened second; destroyed first
    ASSERT_FALSE(ec) << "Failed to open DataLog at " << wpilog_path << ": "
                     << ec.message();
    NT_DataLogger nt_logger =
        nt_instance.StartEntryDataLog(data_log, "", "NT:");

    const std::vector<camera::camera_constant_t> cameras = {
        cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                              extrinsics_path)};
    auto pipeline = MakeLocalizationPipeline(
        kEncodedJpegPath, 20283378, cameras,
        std::make_unique<camera::OpenCVDecodeNode>("pipeline_test_cpu"));
    auto sender = std::make_unique<localization::NetworkTableSender>(
        "front", nt_instance);
    std::atomic<bool> done{false};

    pipeline.solver->RegisterCallback(
        [&](localization::position_estimate_t estimate,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context>) {
          sender->Send(estimate, std::move(metadata), nullptr);
          nt_instance.FlushLocal();
          if (estimates_received.fetch_add(1) + 1 >= kTargetEstimates) {
            done = true;
            pipeline.controller->RequestStop();
          }
        });

    RunUntilDone(pipeline, done);

    // Tear down in dependency order before the scope closes.
    sender.reset();
    wpi::nt::NetworkTableInstance::StopEntryDataLog(nt_logger);
    nt_instance.StopLocal();
    wpi::nt::NetworkTableInstance::Destroy(nt_instance);
    // data_log destroyed here (last declared → first destroyed in scope LIFO):
    // its raw_ostream destructor flushes all buffered bytes to disk.
  }

  EXPECT_GE(estimates_received.load(), kTargetEstimates)
      << "Timed out before reaching " << kTargetEstimates << " estimates";
  ASSERT_TRUE(std::filesystem::exists(wpilog_path))
      << "Expected wpilog at " << wpilog_path;
  EXPECT_GT(std::filesystem::file_size(wpilog_path), 1000u)
      << "wpilog at " << wpilog_path << " is suspiciously small";
}

// Drives every frame in a directory of recorded log images (e.g. a real
// match's camera roll) through the CPU pipeline -- OpenCVDecodeNode,
// OpenCVApriltagDetectorNode, UnambiguousSolverNode -- and logs every
// estimate via NetworkTableSender/DataLogWriter, producing a wpilog whose
// size actually reflects real, varied field data rather than the same frame
// replayed a handful of times. Opt-in: set COS_PIPELINE_LOG_DIR to a
// directory of "<seconds>.<micros>.jpg" frames to run it.
TEST(LocalizationIntegrationTest, WritesWpilogFromRecordedLogOnCpu) {
  const char* log_dir_env = std::getenv("COS_PIPELINE_LOG_DIR");
  if (log_dir_env == nullptr) {
    GTEST_SKIP() << "Set COS_PIPELINE_LOG_DIR to a directory of recorded "
                 << "\"<seconds>.<micros>.jpg\" frames to run this test.";
  }
  const std::filesystem::path log_dir(log_dir_env);
  ASSERT_TRUE(std::filesystem::is_directory(log_dir)) << log_dir;
  const std::vector<std::pair<std::filesystem::path, unsigned long>> frames =
      ListSortedFrames(log_dir);
  ASSERT_FALSE(frames.empty()) << "No .jpg frames found in " << log_dir;

  const auto wpilog_dir = std::filesystem::path(COS_BINARY_DIR) / "wpilogs";
  std::filesystem::create_directories(wpilog_dir);
  const std::string wpilog_path =
      (wpilog_dir / "WritesWpilogFromRecordedLogOnCpu.wpilog").string();
  std::filesystem::remove(wpilog_path);

  const auto tmp = std::filesystem::temp_directory_path();
  const std::string intrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "recorded_log_intrinsics.json", cos_test::testing::IntrinsicsJson());
  const std::string extrinsics_path = cos_test::testing::WriteJsonFile(
      tmp / "recorded_log_extrinsics.json", cos_test::testing::ExtrinsicsJson());
  const std::vector<camera::camera_constant_t> cameras = {
      cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                            extrinsics_path)};

  std::atomic<int> estimates_received{0};

  // Same teardown-order requirement as the tests above: everything touching
  // the DataLog/NT instance must be destroyed before this scope closes.
  {
    auto nt_instance = wpi::nt::NetworkTableInstance::Create();
    nt_instance.StartLocal();
    std::error_code ec;
    wpi::log::DataLogWriter data_log(wpilog_path, ec);
    ASSERT_FALSE(ec) << "Failed to open DataLog at " << wpilog_path << ": "
                     << ec.message();
    NT_DataLogger nt_logger =
        nt_instance.StartEntryDataLog(data_log, "", "NT:");

    camera::OpenCVDecodeNode decoder("recorded_log_pipeline_cpu");
    apriltag::OpenCVApriltagDetectorNode detector(
        cos_test::testing::IntrinsicsJson());
    localization::UnambiguousSolverNode solver(cameras);
    auto sender = std::make_unique<localization::NetworkTableSender>(
        "front", nt_instance);

    // Frames are fed one at a time, waiting for this frame's detection pass
    // to finish before starting the next, so every frame in the log is
    // actually processed (rather than the LoopController's usual
    // latest-frame-wins behavior). The wait is keyed off the detector's
    // callback rather than the solver's: UnambiguousSolverNode only invokes
    // its callback when it finds a valid pose (see SolveAndReset), and most
    // real-log frames don't have a tag in view, so waiting on the solver
    // would hang forever on those frames.
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    bool frame_done = false;

    decoder.RegisterCallback(
        [&detector](std::shared_ptr<camera::DecodedJpegNvBuffer> frame,
                   control_loops::MetaDataList metadata,
                   std::shared_ptr<control_loops::Context> ctx) {
          detector.Detect(std::move(frame), std::move(metadata),
                          std::move(ctx));
        });
    detector.RegisterCallback(
        [&](std::shared_ptr<std::vector<apriltag::tag_detection_t>>
                detections,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> ctx) {
          solver.Accumulate(detections, metadata, ctx);
          {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame_done = true;
          }
          frame_cv.notify_one();
        });

    solver.RegisterCallback(
        [&](localization::position_estimate_t estimate,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context>) {
          sender->Send(estimate, std::move(metadata), nullptr);
          nt_instance.FlushLocal();
          estimates_received.fetch_add(1);
        });

    for (const auto& [frame_path, timestamp] : frames) {
      camera::DiskCamera camera(frame_path, timestamp);
      camera.RegisterCallback(
          [&](std::shared_ptr<camera::JpegBuffer> jpeg, unsigned long ts) {
            decoder.Decode(jpeg, {{.camera_idx = 0, .timestamp = ts}},
                           nullptr);
          });

      {
        std::lock_guard<std::mutex> lock(frame_mutex);
        frame_done = false;
      }
      camera.Start();

      std::unique_lock<std::mutex> lock(frame_mutex);
      const bool completed = frame_cv.wait_for(
          lock, std::chrono::seconds(10), [&] { return frame_done; });
      ASSERT_TRUE(completed) << "Timed out processing frame " << frame_path;
    }

    sender.reset();
    wpi::nt::NetworkTableInstance::StopEntryDataLog(nt_logger);
    nt_instance.StopLocal();
    wpi::nt::NetworkTableInstance::Destroy(nt_instance);
  }

  // Not every frame in a real log has a tag in view (or yields a valid,
  // on-field pose), so this only requires that most of them did -- not that
  // every single frame produced an estimate.
  EXPECT_GT(estimates_received.load(), static_cast<int>(frames.size()) / 2)
      << "Too few estimates (" << estimates_received.load() << "/"
      << frames.size() << ") -- pipeline may not be detecting tags";
  ASSERT_TRUE(std::filesystem::exists(wpilog_path))
      << "Expected wpilog at " << wpilog_path;
  EXPECT_GT(std::filesystem::file_size(wpilog_path), 10'000u)
      << "wpilog at " << wpilog_path << " is suspiciously small";
}

// -----------------------------------------------------------------------
// Gamepiece pipeline
// -----------------------------------------------------------------------

TEST(GamepiecePipelineTest, ProducesDetectionsFromDiskFrames) {
  if (!cos_test::testing::CudaDeviceCount().has_value()) {
    GTEST_SKIP() << "Gamepiece pipeline requires a CUDA-capable GPU";
  }
  if (!HasNvidiaJpegRuntime()) {
    GTEST_SKIP() << "Gamepiece pipeline requires NVIDIA's libjpeg runtime; "
                 << "linked libjpeg is " << LinkedJpegLibraryPath();
  }
  if (!std::filesystem::exists(kGamepieceModelPath)) {
    GTEST_SKIP() << "Gamepiece model not found at " << kGamepieceModelPath;
  }

  ASSERT_TRUE(std::filesystem::exists(kEncodedJpegPath))
      << "Missing MJPEG camera fixture: " << kEncodedJpegPath;

  std::unique_ptr<camera::ICamera> camera =
      std::make_unique<camera::DiskCamera>(kEncodedJpegPath, 20283378);
  auto localization_controller =
      std::make_shared<control_loops::LocalizationLoopController>(1);
  auto decoder =
      std::make_unique<camera::NvjpegDecodeNode>("gamepiece_pipeline_test");
  auto gamepiece_controller =
      std::make_shared<control_loops::GamepieceLoopController>(1);
  auto yolo = std::make_unique<gamepiece::YoloNode>(
      kGamepieceModelPath, std::vector<std::string>{"coral", "algae"},
      cos_test::testing::IntrinsicsJson(), cos_test::testing::ExtrinsicsJson());

  std::atomic<bool> detection_received{false};
  yolo->RegisterCallback([&](std::vector<gamepiece::gamepiece_detection_t>,
                             control_loops::MetaDataList,
                             std::shared_ptr<control_loops::Context>) {
    detection_received = true;
    localization_controller->RequestStop();
    gamepiece_controller->RequestStop();
  });

  decoder->RegisterCallback(
      [gamepiece_controller](
          std::shared_ptr<camera::DecodedJpegNvBuffer> decoded,
          control_loops::MetaDataList metadata,
          std::shared_ptr<control_loops::Context>) {
        const unsigned long timestamp =
            metadata.empty() ? 0 : metadata.front().timestamp;
        gamepiece_controller->ReceiveFrame(0, std::move(decoded), timestamp);
      });

  auto* decoder_ptr = decoder.get();
  localization_controller->RegisterIterationCallback(
      0, [decoder_ptr](std::shared_ptr<camera::JpegBuffer> jpeg,
                       control_loops::MetaDataList metadata,
                       std::shared_ptr<control_loops::Context> ctx) {
        decoder_ptr->Decode(std::move(jpeg), std::move(metadata),
                            std::move(ctx));
      });

  auto* yolo_ptr = yolo.get();
  gamepiece_controller->RegisterIterationCallback(
      0, [yolo_ptr](std::shared_ptr<camera::DecodedJpegNvBuffer> frame,
                    control_loops::MetaDataList metadata,
                    std::shared_ptr<control_loops::Context> ctx) {
        yolo_ptr->Detect(frame, std::move(metadata), std::move(ctx));
      });

  camera->RegisterCallback(
      [localization_controller](std::shared_ptr<camera::JpegBuffer> frame,
                                unsigned long timestamp) {
        localization_controller->ReceiveFrame(0, std::move(frame), timestamp);
      });

  camera->Start();
  {
    std::jthread localization_thread(
        [&localization_controller] { localization_controller->Run(); });
    std::jthread gamepiece_thread(
        [&gamepiece_controller] { gamepiece_controller->Run(); });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!detection_received.load() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    localization_controller->RequestStop();
    gamepiece_controller->RequestStop();
  }

  EXPECT_TRUE(detection_received.load());
}

}  // namespace
