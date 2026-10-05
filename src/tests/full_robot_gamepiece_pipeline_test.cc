#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/cpu_apriltag_detector_node.h"
#include "camera/camera_config.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/jpeg_buffer.h"
#include "camera/cpu_decode_node.h"
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

ABSL_FLAG(std::string, image_folder, "/cos-logs/gamepiece",  // NOLINT
          "Directory containing timestamped JPEG frames");
ABSL_FLAG(std::string, model_path,  // NOLINT
          "/root/gamepiece_models/best_nms_gray.engine",  // NOLINT
          "TensorRT engine with embedded NMS output");
ABSL_FLAG(std::string, annotation_dir, "/root/gamepiece_logs/annotated",  // NOLINT
          "Directory for frames annotated with TensorRT detections");
ABSL_FLAG(int, image_width, 1920,  // NOLINT
          "Width of the replay JPEGs");
ABSL_FLAG(int, image_height, 1080,  // NOLINT
          "Height of the replay JPEGs");
ABSL_FLAG(int, timeout_seconds, 300,  // NOLINT
          "Maximum time to wait for the replay to complete");

namespace {

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

auto DummyIntrinsics() -> nlohmann::json {
  return {{"cx", 640.0}, {"cy", 400.0}, {"fx", 905.0}, {"fy", 905.0},
          {"k1", 0.0},   {"k2", 0.0},   {"k3", 0.0},   {"p1", 0.0},
          {"p2", 0.0}};
}

auto DummyExtrinsics() -> nlohmann::json {
  return {{"translation_x", 0.0}, {"translation_y", 0.0},
          {"translation_z", 1.0}, {"rotation_x", 0.0},
          {"rotation_y", 0.35},   {"rotation_z", 0.0}};
}

class TemporaryCalibration final {
 public:
  TemporaryCalibration(int width, int height) {
    directory_ = fs::temp_directory_path() /
                 ("cos-gamepiece-calibration-" +
                  std::to_string(static_cast<long long>(getpid())));
    detector_config_path_ = directory_ / "detector.json";
    CHECK(fs::create_directory(directory_))
        << "Unable to create temporary calibration directory: "
        << directory_;
    WriteJson(detector_config_path_, {{"width", width}, {"height", height},
                                    {"intrinsics", DummyIntrinsics()},
                                    {"extrinsics", DummyExtrinsics()}});
  }

  ~TemporaryCalibration() {
    std::error_code error;
    fs::remove_all(directory_, error);
    if (error) {
      LOG(WARNING) << "Unable to remove temporary calibration directory: "
                   << directory_ << ": " << error.message();
    }
  }

  TemporaryCalibration(const TemporaryCalibration&) = delete;
  auto operator=(const TemporaryCalibration&) -> TemporaryCalibration& = delete;

  [[nodiscard]] auto DetectorConfigPath() const -> const fs::path& {
    return detector_config_path_;
  }

 private:
  void WriteJson(const fs::path& path, const nlohmann::json& value) {
    std::ofstream stream(path);
    CHECK(stream.is_open()) << "Unable to write temporary calibration: "
                            << path;
    stream << value.dump(2) << '\n';
  }

  fs::path directory_;
  fs::path detector_config_path_;
};

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
      label << "class_id=" << detection.class_id << " confidence="
            << std::fixed << std::setprecision(2) << detection.confidence;
      const cv::Point origin(detection.bounds.x,
                             std::max(24, detection.bounds.y - 8));
      cv::putText(annotated, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    std::scoped_lock lock(output_mutex_);
    nlohmann::json record = {{"frame_index", frame_index_},
                             {"detections", nlohmann::json::array()}};
    for (const auto& detection : detections) {
      record["detections"].push_back(
          {{"class_id", detection.class_id},
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

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const fs::path image_folder = absl::GetFlag(FLAGS_image_folder);
  const fs::path model_path = absl::GetFlag(FLAGS_model_path);
  const fs::path annotation_dir = absl::GetFlag(FLAGS_annotation_dir);
  CHECK(fs::is_regular_file(model_path)) << "Missing engine: " << model_path;
  CHECK_GT(absl::GetFlag(FLAGS_image_width), 0);
  CHECK_GT(absl::GetFlag(FLAGS_image_height), 0);
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const size_t input_frames = CountFrames(image_folder);
  const double replay_offset =
      camera::GetEarliestTimestamp(image_folder.string());
  TemporaryCalibration calibration(absl::GetFlag(FLAGS_image_width),
                                   absl::GetFlag(FLAGS_image_height));

  constexpr std::string_view kJpegChannel = "gamepiece/jpeg";
  constexpr std::string_view kLocalizationDecodedChannel =
      "localization/decoded";
  constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
  constexpr std::string_view kAprilTagChannel =
      "localization/gamepiece_camera_detection_batch";
  constexpr std::string_view kPositionChannel = "localization/position";
  constexpr std::string_view kDetectionChannel = "gamepiece/detections";

  control_loop::ThreadPool thread_pool(3);
  control_loop::ControlLoop localization_loop(1ms);
  auto localization_decoder = std::make_shared<camera::CpuJpegDecodeNode>(
      kJpegChannel, kLocalizationDecodedChannel, thread_pool);
  auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
      kJpegChannel, kDecodedChannel, NVJPEG_OUTPUT_Y, thread_pool);
  auto apriltag_detector =
      std::make_shared<apriltag::CpuApriltagDetectorNode>(
          kLocalizationDecodedChannel, kAprilTagChannel,
          calibration.DetectorConfigPath().string(),
          thread_pool);

  auto solver = std::make_shared<localization::UnambiguousSolverNode>(
      kPositionChannel);
  solver->AddCamera(kAprilTagChannel,
                    camera::Intrinsics{calibration.DetectorConfigPath()},
                    camera::Extrinsics{calibration.DetectorConfigPath()},
                    localization_loop);

  DetectionAnnotations annotations(annotation_dir);
  auto gamepiece_node = std::make_shared<gamepiece::YoloNode>(
      kDecodedChannel, kDetectionChannel, model_path.string(),
      std::vector<std::string>{}, thread_pool);

  std::atomic<size_t> encoded_frames = 0;
  std::atomic<size_t> decoded_frames = 0;
  std::atomic<size_t> localization_detection_batches = 0;
  std::atomic<size_t> localization_callbacks = 0;
  std::atomic<size_t> gamepiece_batches = 0;
  std::atomic<size_t> total_detections = 0;
  std::atomic<size_t> handoff_frames = 0;
  std::atomic<size_t> unexpected_position_messages = 0;
  std::mutex completion_mutex;
  std::condition_variable completion;
  double last_encoded_timestamp = -INFINITY;
  double last_gamepiece_timestamp = -INFINITY;
  size_t nonmonotonic_frames = 0;

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
        }
      });
  solver->RegisterCallback([&](const control_loop::Context& context) -> void {
    if (context->GetMessage<apriltag::TagDetections>(kAprilTagChannel) !=
        nullptr) {
      ++localization_callbacks;
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
        CHECK(!context->Exists(std::string(kLocalizationDecodedChannel)));
        CHECK(!context->Exists(std::string(kAprilTagChannel)));
        if (context->Exists(std::string(kPositionChannel))) {
          ++unexpected_position_messages;
        }
        total_detections += detections->value.size();
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
  localization_loop.RegisterNode(localization_decoder);
  localization_loop.RegisterNode(decoder);
  localization_loop.RegisterNode(apriltag_detector);
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
          return stop::StopRequested() && encoded_frames.load() > 0U &&
                 last_gamepiece_timestamp == last_encoded_timestamp &&
                 decoded_frames.load() == encoded_frames.load() &&
                 localization_callbacks.load() == encoded_frames.load();
        });
  }

  localization_loop.Stop();
  // Stop both schedulers before draining their shared worker pool while all
  // nodes and callbacks are still alive.
  gamepiece_loop.Stop();
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

  size_t annotation_count = 0;
  for (const auto& entry : fs::directory_iterator(annotation_dir)) {
    if (entry.is_regular_file() && IsTimestampedJpeg(entry.path())) {
      ++annotation_count;
    }
  }
  CHECK_EQ(annotation_count, gamepiece_batches.load());
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
            << " annotations=" << annotation_count;
  return 0;
}
