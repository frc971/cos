#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/camera_config.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/jpeg_buffer.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/context_handoff_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/yolo_node.h"
#include "localization/unambiguous_solver_node.h"
#include "utils/stop.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

ABSL_FLAG(std::string, image_folder, "/cos-logs/second_bot/chezychamps/front",  // NOLINT
          "Directory containing timestamped JPEG frames");
ABSL_FLAG(std::string, model_path,  // NOLINT
          "/root/gamepiece_models/yolo11n_nms_gray.engine",  // NOLINT
          "TensorRT engine with embedded NMS output");
ABSL_FLAG(std::string, annotation_dir, "/root/gamepiece_logs/annotated",  // NOLINT
          "Directory for frames annotated with TensorRT detections");
ABSL_FLAG(std::string, constants_dir, "/root/constants/second_bot",  // NOLINT
          "Calibration directory containing front/left/right_camera.json");
ABSL_FLAG(std::string, localization_log_path, "",  // NOLINT
          "Optional replay root for additional left and right localization cameras");
ABSL_FLAG(std::string, class_names, "",  // NOLINT
          "Optional text file with one model class name per line");
ABSL_FLAG(int, min_person_detections, 1,  // NOLINT
          "Minimum COCO person (class 0) detections required");
ABSL_FLAG(int, min_localization_poses, 0,  // NOLINT
          "Minimum finite localization poses required; no ground truth comparison");
ABSL_FLAG(int, image_width, 1280,  // NOLINT
          "Width of the replay JPEGs");
ABSL_FLAG(int, image_height, 800,  // NOLINT
          "Height of the replay JPEGs");
ABSL_FLAG(int, timeout_seconds, 300,  // NOLINT
          "Maximum time to wait for the replay to complete");

namespace {

struct CameraReplayStats {
  std::atomic<size_t> encoded = 0;
  std::atomic<size_t> decoded = 0;
  std::atomic<size_t> detection_batches = 0;
};

auto IsTimestampedJpeg(const fs::path& path) -> bool {
  if (!path.has_extension()) {
    return false;
  }
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(),
                         [](unsigned char character) -> char {
                           return static_cast<char>(std::tolower(character));
                         });
  if (extension != ".jpg" && extension != ".jpeg") {
    return false;
  }
  try {
    size_t parsed_characters = 0;
    const std::string stem = path.stem().string();
    const double timestamp = std::stod(stem, &parsed_characters);
    return parsed_characters == stem.size() && std::isfinite(timestamp);
  } catch (const std::exception&) {
    return false;
  }
}

auto CountFrames(const fs::path& directory) -> size_t {
  CHECK(fs::is_directory(directory)) << "Missing image folder: " << directory;
  const size_t count = std::ranges::count_if(
      fs::directory_iterator(directory), [](const fs::directory_entry& entry) -> bool {
        return entry.is_regular_file() && IsTimestampedJpeg(entry.path());
      });
  CHECK_GT(count, 0U) << "No timestamped JPEGs found in " << directory;
  return count;
}

class DetectionAnnotations final {
 public:
  explicit DetectionAnnotations(fs::path output_directory)
      : output_directory_(std::move(output_directory)) {
    CHECK(!fs::exists(output_directory_) || fs::is_empty(output_directory_))
        << "Annotation directory must be new or empty: " << output_directory_;
    fs::create_directories(output_directory_);
    detections_log_.open(output_directory_ / "detections.jsonl");
    CHECK(detections_log_.is_open());
  }

  void Write(const camera::DecodedJpegBuffer& frame,
             const std::vector<gamepiece::labeled_bounding_box_t>& detections) {
    const cv::cuda::GpuMat image(frame.height, frame.width, CV_8UC1,
                               frame.destination.channel[0], frame.stride);
    for (const auto& detection : detections) {
      CHECK_GE(detection.class_id, 0);
      CHECK(std::isfinite(detection.confidence));
      CHECK_GE(detection.confidence, 0.0F);
      CHECK_LE(detection.confidence, 1.0F);
      CHECK(!detection.bounds.empty());
      CHECK_EQ(detection.bounds & cv::Rect(0, 0, image.cols, image.rows),
               detection.bounds);
    }
    cv::Mat grayscale;
    image.download(grayscale);
    cv::Mat annotated;
    cv::cvtColor(grayscale, annotated, cv::COLOR_GRAY2BGR);
    for (const auto& detection : detections) {
      cv::rectangle(annotated, detection.bounds, cv::Scalar(0, 255, 0), 3);
      std::ostringstream label;
      label << detection.label << " class_id=" << detection.class_id << " confidence="
            << std::fixed << std::setprecision(2) << detection.confidence;
      const cv::Point origin(detection.bounds.x,
                             std::max(24, detection.bounds.y - 8));
      cv::putText(annotated, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    std::scoped_lock lock(output_mutex_);
    nlohmann::json record = {{"frame_index", frame_index_},
                             {"timestamp", frame.timestamp},
                             {"detections", nlohmann::json::array()}};
    for (const auto& detection : detections) {
      record["detections"].push_back(
          {{"class_id", detection.class_id},
           {"label", detection.label},
           {"confidence", detection.confidence},
           {"bounds", {detection.bounds.x, detection.bounds.y,
                        detection.bounds.width, detection.bounds.height}}});
    }
    detections_log_ << record.dump() << '\n' << std::flush;
    CHECK(detections_log_.good());
    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6) << frame_index_++ << ".jpg";
    CHECK(cv::imwrite((output_directory_ / filename.str()).string(),
                      annotated));
  }

 private:
  fs::path output_directory_;
  std::ofstream detections_log_;
  std::mutex output_mutex_;
  size_t frame_index_ = 0;
};

auto RunReplay() -> int {

  const fs::path image_folder = absl::GetFlag(FLAGS_image_folder);
  const fs::path model_path = absl::GetFlag(FLAGS_model_path);
  const fs::path annotation_dir = absl::GetFlag(FLAGS_annotation_dir);
  CHECK(fs::is_regular_file(model_path)) << "Missing engine: " << model_path;
  CHECK_GT(absl::GetFlag(FLAGS_image_width), 0);
  CHECK_GT(absl::GetFlag(FLAGS_image_height), 0);
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const size_t input_frames = CountFrames(image_folder);
  std::vector<std::string> replay_paths{image_folder.string()};
  const fs::path localization_log_path =
      absl::GetFlag(FLAGS_localization_log_path);
  if (!localization_log_path.empty()) {
    replay_paths.push_back((localization_log_path / "left").string());
    replay_paths.push_back((localization_log_path / "right").string());
  }
  const double replay_offset = camera::GetEarliestTimestamp(replay_paths);
  const fs::path constants_dir = absl::GetFlag(FLAGS_constants_dir);
  const fs::path config_path = constants_dir / "front_camera.json";
  CHECK(fs::is_regular_file(config_path));
  CHECK_GE(absl::GetFlag(FLAGS_min_person_detections), 0);
  CHECK_GE(absl::GetFlag(FLAGS_min_localization_poses), 0);
  std::vector<std::string> class_names;
  if (!absl::GetFlag(FLAGS_class_names).empty()) {
    std::ifstream names(absl::GetFlag(FLAGS_class_names));
    CHECK(names.is_open());
    for (std::string name; std::getline(names, name);) {
      class_names.push_back(std::move(name));
    }
    CHECK(!class_names.empty());
    CHECK_EQ(class_names.front(), "person");
  }

  constexpr std::string_view kJpegChannel = "gamepiece/jpeg";
  constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
  constexpr std::string_view kAprilTagChannel =
      "localization/gamepiece_camera_detection_batch";
  constexpr std::string_view kPositionChannel = "localization/position";
  constexpr std::string_view kDetectionChannel = "gamepiece/detections";

  control_loop::ThreadPool thread_pool(6);
  control_loop::ControlLoop localization_loop(1ms);
  auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
      kJpegChannel, kDecodedChannel, NVJPEG_OUTPUT_Y, thread_pool);
  auto apriltag_detector =
      std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
          kDecodedChannel, kAprilTagChannel,
          config_path.string(),
          thread_pool);

  auto solver = std::make_shared<localization::UnambiguousSolverNode>(
      kPositionChannel);
  solver->AddCamera(kAprilTagChannel,
                    camera::Intrinsics{config_path},
                    camera::Extrinsics{config_path},
                    localization_loop);

  DetectionAnnotations annotations(annotation_dir);
  auto gamepiece_node = std::make_shared<gamepiece::YoloNode>(
      kDecodedChannel, kDetectionChannel, model_path.string(),
      class_names, thread_pool);

  std::atomic<size_t> encoded_frames = 0;
  std::atomic<size_t> decoded_frames = 0;
  std::atomic<size_t> localization_detection_batches = 0;
  std::atomic<size_t> localization_callbacks = 0;
  std::atomic<size_t> gamepiece_batches = 0;
  std::atomic<size_t> total_detections = 0;
  std::atomic<size_t> person_detections = 0;
  std::atomic<size_t> localization_poses = 0;
  std::atomic<size_t> detected_tags = 0;
  std::atomic<size_t> handoff_frames = 0;
  std::atomic<size_t> unexpected_position_messages = 0;
  std::array<CameraReplayStats, 2> additional_camera_stats;
  std::array<size_t, 2> additional_input_frames{};
  std::mutex completion_mutex;
  std::condition_variable completion;
  double last_encoded_timestamp = -INFINITY;
  double last_gamepiece_timestamp = -INFINITY;
  size_t nonmonotonic_frames = 0;
  std::uint64_t last_localization_context = 0;

  control_loop::RioClock::EnableSimulation();
  auto camera = std::make_shared<camera::UVCDiskCameraNode>(
      image_folder.string(), kJpegChannel, replay_offset);
  camera->RegisterCallback([&](const control_loop::Context& context) -> void {
    const auto* jpeg = context->GetMessage<camera::JpegBuffer>(kJpegChannel);
    if (jpeg != nullptr && jpeg->ptr != nullptr) {
      std::scoped_lock lock(completion_mutex);
      ++encoded_frames;
      last_encoded_timestamp = jpeg->timestamp;
    }
  });
  decoder->RegisterCallback([&](const control_loop::Context& context) -> void {
    const auto* decoded =
        context->GetMessage<camera::DecodedJpegBuffer>(kDecodedChannel);
    if (decoded != nullptr && decoded->destination.channel[0] != nullptr) {
      CHECK_EQ(decoded->width, absl::GetFlag(FLAGS_image_width));
      CHECK_EQ(decoded->height, absl::GetFlag(FLAGS_image_height));
      ++decoded_frames;
    }
  });
  apriltag_detector->RegisterCallback(
      [&](const control_loop::Context& context) -> void {
        const auto* detections =
            context->GetMessage<apriltag::TagDetections>(kAprilTagChannel);
        if (detections != nullptr) {
          ++localization_detection_batches;
          detected_tags += detections->tag_detections.size();
        }
      });
  solver->RegisterCallback([&](const control_loop::Context& context) -> void {
    // Multiple detector publications can become ready before their solver
    // callbacks acquire its lock. Count each context once in that case.
    if (context->id == last_localization_context) {
      return;
    }
    last_localization_context = context->id;
    if (context->GetMessage<apriltag::TagDetections>(kAprilTagChannel) !=
        nullptr) {
      ++localization_callbacks;
    }
    const auto* pose = context->GetMessage<localization::PositionEstimateMessage>(
        kPositionChannel);
    if (pose != nullptr) {
      CHECK(std::isfinite(pose->pose.X().value()));
      CHECK(std::isfinite(pose->pose.Y().value()));
      CHECK(std::isfinite(pose->pose.Z().value()));
      CHECK(std::isfinite(pose->pose.Rotation().X().value()));
      CHECK(std::isfinite(pose->pose.Rotation().Y().value()));
      CHECK(std::isfinite(pose->pose.Rotation().Z().value()));
      ++localization_poses;
    }
    std::scoped_lock lock(completion_mutex);
    completion.notify_one();
  });
  gamepiece_node->RegisterCallback(
      [&](const control_loop::Context& context) -> void {
        const auto* detections = context->GetMessage<
            control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
            kDetectionChannel);
        if (detections == nullptr) {
          return;
        }
        const auto* frame =
            context->GetMessage<camera::DecodedJpegBuffer>(kDecodedChannel);
        CHECK(frame != nullptr);
        ++handoff_frames;
        CHECK(!context->Exists(std::string(kJpegChannel)));
        CHECK(!context->Exists(std::string(kAprilTagChannel)));
        if (context->Exists(std::string(kPositionChannel))) {
          ++unexpected_position_messages;
        }
        total_detections += detections->value.size();
        for (const auto& detection : detections->value) {
          if (detection.class_id == 0) {
            ++person_detections;
          }
        }
        annotations.Write(*frame, detections->value);
        std::scoped_lock lock(completion_mutex);
        if (frame->timestamp <= last_gamepiece_timestamp) {
          ++nonmonotonic_frames;
        }
        last_gamepiece_timestamp = frame->timestamp;
        const size_t batches = ++gamepiece_batches;
        if (batches % 25 == 0 || batches == input_frames) {
          LOG(INFO) << "Full robot replay progress: " << batches << "/"
                    << input_frames << " gamepiece batches, detections="
                    << total_detections.load();
        }
        completion.notify_one();
      });

  control_loop::ControlLoop gamepiece_loop(20ms);
  gamepiece_loop.RegisterDependencyNode(
      std::make_shared<control_loop::ContextHandoffNode>(decoder));
  gamepiece_loop.RegisterNode(gamepiece_node);
  localization_loop.RegisterDependencyNode(camera);
  localization_loop.RegisterNode(decoder);
  localization_loop.RegisterNode(apriltag_detector);

  // Use the production nvJPEG -> NVIDIA AprilTag -> multitag solver path for
  // all three cameras, while only front-camera images cross into the YOLO loop.
  const std::array<std::string, 2> additional_camera_names{"left", "right"};
  if (!localization_log_path.empty()) {
    for (size_t i = 0; i < additional_camera_names.size(); ++i) {
      const std::string& name = additional_camera_names[i];
      const fs::path folder = localization_log_path / name;
      additional_input_frames[i] = CountFrames(folder);
      const fs::path calibration = constants_dir / (name + "_camera.json");
      CHECK(fs::is_regular_file(calibration));
      const std::string jpeg_channel = "localization/" + name + "/jpeg";
      const std::string decoded_channel = "localization/" + name + "/decoded";
      const std::string detection_channel = "localization/" + name + "/tags";
      auto replay_camera = std::make_shared<camera::UVCDiskCameraNode>(
          folder.string(), jpeg_channel, replay_offset);
      auto replay_decoder = std::make_shared<camera::NvjpegDecodeNode>(
          jpeg_channel, decoded_channel, NVJPEG_OUTPUT_Y, thread_pool);
      auto replay_detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
          decoded_channel, detection_channel, calibration.string(), thread_pool);
      replay_camera->RegisterCallback(
          [&, i, jpeg_channel](const control_loop::Context& context) -> void {
            if (context->GetMessage<camera::JpegBuffer>(jpeg_channel) != nullptr) {
              ++additional_camera_stats[i].encoded;
            }
          });
      replay_decoder->RegisterCallback(
          [&, i, decoded_channel](const control_loop::Context& context) -> void {
            if (context->GetMessage<camera::DecodedJpegBuffer>(decoded_channel) != nullptr) {
              ++additional_camera_stats[i].decoded;
            }
          });
      replay_detector->RegisterCallback(
          [&, i, detection_channel](const control_loop::Context& context) -> void {
            const auto* tags = context->GetMessage<apriltag::TagDetections>(detection_channel);
            if (tags != nullptr) {
              ++additional_camera_stats[i].detection_batches;
              detected_tags += tags->tag_detections.size();
            }
            std::scoped_lock lock(completion_mutex);
            completion.notify_one();
          });
      localization_loop.RegisterDependencyNode(replay_camera);
      localization_loop.RegisterNode(replay_decoder);
      localization_loop.RegisterNode(replay_detector);
      solver->AddCamera(detection_channel, camera::Intrinsics{calibration},
                        camera::Extrinsics{calibration}, localization_loop);
    }
  }
  localization_loop.RegisterNode(solver);

  gamepiece_loop.Start();
  localization_loop.Start();
  LOG(INFO) << "Running full robot replay for " << input_frames
            << " timestamped JPEG frames";

  bool completed = false;
  {
    std::unique_lock lock(completion_mutex);
    completed = completion.wait_for(
        lock, std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)), [&]() -> bool {
          const bool additional_complete = std::ranges::all_of(
              additional_camera_stats, [](const CameraReplayStats& stats) -> bool {
                return stats.encoded.load() == stats.decoded.load() &&
                       stats.encoded.load() == stats.detection_batches.load();
              });
          return stop::StopRequested() && encoded_frames.load() > 0U &&
                 additional_complete &&
                 last_gamepiece_timestamp == last_encoded_timestamp &&
                 decoded_frames.load() == encoded_frames.load() &&
                 localization_callbacks.load() == encoded_frames.load();
        });
  }

  // Keep localization producing handoff notifications until the blocking
  // gamepiece scheduler has stopped, then drain the shared worker pool.
  gamepiece_loop.Stop();
  localization_loop.Stop();
  thread_pool.Shutdown();

  CHECK(completed) << "Timed out after "
                   << absl::GetFlag(FLAGS_timeout_seconds)
                   << " seconds: gamepiece batches=" << gamepiece_batches.load()
                   << "/" << input_frames;
  CHECK_GT(encoded_frames.load(), 0U);
  CHECK_LE(encoded_frames.load(), input_frames);
  CHECK_EQ(decoded_frames.load(), encoded_frames.load());
  CHECK_EQ(localization_detection_batches.load(), encoded_frames.load());
  CHECK_EQ(localization_callbacks.load(), encoded_frames.load());
  CHECK_GT(gamepiece_batches.load(), 0U);
  CHECK_LE(gamepiece_batches.load(), decoded_frames.load());
  CHECK_EQ(handoff_frames.load(), gamepiece_batches.load());
  CHECK_EQ(nonmonotonic_frames, 0U);
  CHECK_EQ(last_gamepiece_timestamp, last_encoded_timestamp);
  CHECK_EQ(unexpected_position_messages.load(), 0U);
  CHECK_GT(total_detections.load(), 0U)
      << "The model produced no bounding boxes";
  CHECK_GE(person_detections.load(),
           static_cast<size_t>(absl::GetFlag(FLAGS_min_person_detections)))
      << "YOLO11n did not detect the required number of people";
  CHECK_GE(localization_poses.load(),
           static_cast<size_t>(absl::GetFlag(FLAGS_min_localization_poses)))
      << "Localization did not produce the required number of finite poses";
  if (!localization_log_path.empty()) {
    for (size_t i = 0; i < additional_camera_stats.size(); ++i) {
      const auto& stats = additional_camera_stats[i];
      CHECK_GT(stats.encoded.load(), 0U) << additional_camera_names[i];
      CHECK_LE(stats.encoded.load(), additional_input_frames[i]);
      CHECK_EQ(stats.decoded.load(), stats.encoded.load());
      CHECK_EQ(stats.detection_batches.load(), stats.encoded.load());
    }
  }

  size_t annotation_count = 0;
  for (const auto& entry : fs::directory_iterator(annotation_dir)) {
    if (entry.is_regular_file() && IsTimestampedJpeg(entry.path())) {
      ++annotation_count;
    }
  }
  CHECK_EQ(annotation_count, gamepiece_batches.load());
  nlohmann::json summary = {
      {"input_frames", input_frames}, {"encoded_frames", encoded_frames.load()},
      {"decoded_frames", decoded_frames.load()},
      {"localization_callbacks", localization_callbacks.load()},
      {"localization_poses", localization_poses.load()},
      {"detected_tags", detected_tags.load()},
      {"gamepiece_batches", gamepiece_batches.load()},
      {"person_detections", person_detections.load()},
      {"total_detections", total_detections.load()},
      {"annotations", annotation_count},
      {"additional_cameras", nlohmann::json::object()}};
  if (!localization_log_path.empty()) {
    for (size_t i = 0; i < additional_camera_stats.size(); ++i) {
      const auto& stats = additional_camera_stats[i];
      summary["additional_cameras"][additional_camera_names[i]] = {
          {"input_frames", additional_input_frames[i]},
          {"encoded_frames", stats.encoded.load()},
          {"decoded_frames", stats.decoded.load()},
          {"detection_batches", stats.detection_batches.load()}};
    }
  }
  std::ofstream summary_file(annotation_dir / "summary.json");
  summary_file << summary.dump(2) << '\n';
  summary_file.flush();
  CHECK(summary_file.good());
  LOG(INFO) << "Full robot replay complete: encoded="
            << encoded_frames.load() << " decoded=" << decoded_frames.load()
            << " localization_batches="
            << localization_detection_batches.load()
            << " localization_callbacks=" << localization_callbacks.load()
            << " gamepiece_batches=" << gamepiece_batches.load()
            << " camera_skipped_frames=" << input_frames - encoded_frames.load()
            << " gamepiece_skipped_frames="
            << decoded_frames.load() - gamepiece_batches.load()
            << " detections=" << total_detections.load()
            << " person_detections=" << person_detections.load()
            << " detected_tags=" << detected_tags.load()
            << " localization_poses=" << localization_poses.load()
            << " annotations=" << annotation_count;
  return 0;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  const int result = RunReplay();
  LOG(INFO) << "Replay nodes destroyed successfully";
  // Keep node destruction inside RunReplay so failures there remain visible.
  // Match the existing Orin localization tests after all workers, nodes, and
  // output streams are cleaned up: NVIDIA's process-exit finalizers conflict
  // when VPI and CUDA inference have both been used in one process.
  std::fflush(nullptr);
  std::_Exit(result);
}
