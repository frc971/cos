#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/uvc_disk_camera_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/yolo_node.h"
#include "control_loop/rio_clock.h"
#include "utils/stop.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

ABSL_FLAG(std::string, image_folder, "",  // NOLINT
          "Directory containing timestamped JPEG frames");
ABSL_FLAG(std::string, model_path, "",  // NOLINT
          "TensorRT engine with embedded NMS output");
ABSL_FLAG(std::string, class_names, "",  // NOLINT
          "Text file containing one class name per line");
ABSL_FLAG(int, timeout_seconds, 30,  // NOLINT
          "Maximum time to wait for the replay to complete");
ABSL_FLAG(std::string, annotation_dir, "",  // NOLINT
          "Optional directory for frames annotated with TensorRT detections");

namespace {

class DetectionAnnotations final {
 public:
  explicit DetectionAnnotations(fs::path output_directory)
      : output_directory_(std::move(output_directory)) {
    fs::create_directories(output_directory_);
  }

  void Write(const camera::DecodedJpegBuffer& frame,
             const std::vector<gamepiece::labeled_bounding_box_t>& detections) {
    const cv::cuda::GpuMat image(frame.height, frame.width, CV_8UC1,
                               frame.destination.channel[0], frame.stride);

    cv::Mat grayscale;
    image.download(grayscale);
    cv::Mat annotated;
    cv::cvtColor(grayscale, annotated, cv::COLOR_GRAY2BGR);
    for (const auto& detection : detections) {
      cv::rectangle(annotated, detection.bounds, cv::Scalar(0, 255, 0), 3);
      std::ostringstream label;
      label << "TRT " << detection.label << " " << std::fixed
            << std::setprecision(2) << detection.confidence;
      const cv::Point origin(detection.bounds.x,
                             std::max(24, detection.bounds.y - 8));
      cv::putText(annotated, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6) << frame_index_++ << ".jpg";
    CHECK(cv::imwrite((output_directory_ / filename.str()).string(), annotated));
  }

 private:
  fs::path output_directory_;
  size_t frame_index_ = 0;
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
    std::stod(stem, &parsed_characters);
    return parsed_characters == stem.size();
  } catch (const std::exception&) {
    return false;
  }
}

auto CountFrames(const fs::path& directory) -> size_t {
  CHECK(fs::is_directory(directory)) << "Missing image folder: " << directory;
  return std::ranges::count_if(
      fs::directory_iterator(directory), [](const fs::directory_entry& entry) -> bool {
        return entry.is_regular_file() && IsTimestampedJpeg(entry.path());
      });
}

auto LoadClassNames(const fs::path& path) -> std::vector<std::string> {
  std::ifstream stream(path);
  CHECK(stream.is_open()) << "Unable to open class names: " << path;
  std::vector<std::string> names;
  for (std::string name; std::getline(stream, name);) {
    if (!name.empty()) {
      names.push_back(std::move(name));
    }
  }
  CHECK(!names.empty()) << "No class names found in " << path;
  return names;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const fs::path image_folder = absl::GetFlag(FLAGS_image_folder);
  const fs::path model_path = absl::GetFlag(FLAGS_model_path);
  const fs::path class_names_path = absl::GetFlag(FLAGS_class_names);
  CHECK(!image_folder.empty());
  CHECK(fs::is_regular_file(model_path)) << "Missing engine: " << model_path;
  CHECK(fs::is_regular_file(class_names_path))
      << "Missing class-name file: " << class_names_path;
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const size_t expected_frames = CountFrames(image_folder);
  CHECK_GT(expected_frames, 0U);
  const std::vector<std::string> class_names =
      LoadClassNames(class_names_path);

  constexpr std::string_view kJpegChannel = "gamepiece/jpeg";
  constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
  constexpr std::string_view kDetectionChannel = "gamepiece/detections";

  control_loop::ThreadPool thread_pool(2);
  control_loop::ControlLoop control_loop(1ms);
  control_loop::RioClock::EnableSimulation();
  auto camera = std::make_shared<camera::UVCDiskCameraNode>(
      image_folder.string(), kJpegChannel,
      camera::GetEarliestTimestamp(image_folder.string()));
  auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
      kJpegChannel, kDecodedChannel, NVJPEG_OUTPUT_Y, thread_pool);
  const fs::path annotation_dir = absl::GetFlag(FLAGS_annotation_dir);
  std::unique_ptr<DetectionAnnotations> annotations;
  if (!annotation_dir.empty()) {
    annotations = std::make_unique<DetectionAnnotations>(annotation_dir);
  }
  auto detector = std::make_shared<gamepiece::YoloNode>(
      kDecodedChannel, kDetectionChannel, model_path.string(), class_names,
      thread_pool);

  std::atomic<size_t> encoded_frames = 0;
  std::atomic<size_t> decoded_frames = 0;
  std::atomic<size_t> detection_batches = 0;
  std::atomic<size_t> total_detections = 0;
  std::atomic<size_t> person_detections = 0;
  std::mutex completion_mutex;
  std::condition_variable completion;

  camera->RegisterCallback([&](const control_loop::Context& context) -> void {
    std::scoped_lock lock(completion_mutex);
    const auto* jpeg = context->GetMessage<camera::JpegBuffer>(kJpegChannel);
    if (jpeg != nullptr && jpeg->ptr != nullptr) {
      ++encoded_frames;
    }
    completion.notify_one();
  });
  decoder->RegisterCallback([&](const control_loop::Context& context) -> void {
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(kDecodedChannel);
    if (frame != nullptr && frame->destination.channel[0] != nullptr) {
      ++decoded_frames;
    }
  });
  detector->RegisterCallback([&](const control_loop::Context& context) -> void {
    std::scoped_lock lock(completion_mutex);
    const auto* detections = context->GetMessage<
        control_loop::ValueMessage<gamepiece::bounding_box_detections_t>>(
        kDetectionChannel);
    if (detections == nullptr) {
      return;
    }
    if (annotations != nullptr) {
      const auto* frame = context->GetMessage<camera::DecodedJpegBuffer>(
          kDecodedChannel);
      CHECK(frame != nullptr);
      annotations->Write(*frame, detections->value);
    }
    total_detections += detections->value.size();
    for (const auto& detection : detections->value) {
      if (detection.class_id == 0) {
        ++person_detections;
      }
    }
    const size_t batches = ++detection_batches;
    if (batches % 25 == 0 || batches == expected_frames) {
      LOG(INFO) << "Gamepiece replay progress: " << batches << "/"
                << expected_frames << " batches, detections="
                << total_detections.load();
    }
    completion.notify_one();
  });

  control_loop.RegisterDependencyNode(camera);
  control_loop.RegisterNode(decoder);
  control_loop.RegisterNode(detector);
  control_loop.Start();
  LOG(INFO) << "Running gamepiece replay for " << expected_frames
            << " timestamped JPEG frames";

  bool completed = false;
  {
    std::unique_lock lock(completion_mutex);
    completed = completion.wait_for(
        lock, std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)), [&]() -> bool {
          return stop::StopRequested() && encoded_frames.load() > 0U &&
                 detection_batches.load() == encoded_frames.load();
        });
  }

  control_loop.Stop();
  thread_pool.Shutdown();

  CHECK(completed) << "Timed out after " << absl::GetFlag(FLAGS_timeout_seconds)
                   << " seconds";
  CHECK_GT(encoded_frames.load(), 0U);
  CHECK_LE(encoded_frames.load(), expected_frames);
  CHECK_EQ(decoded_frames.load(), encoded_frames.load());
  CHECK_EQ(detection_batches.load(), encoded_frames.load());
  CHECK_GT(total_detections.load(), 0U);
  CHECK_GT(person_detections.load(), 0U);

  LOG(INFO) << "Full gamepiece replay complete: encoded="
            << encoded_frames.load() << " decoded=" << decoded_frames.load()
            << " detection_batches=" << detection_batches.load()
            << " detections=" << total_detections.load()
            << " person_detections=" << person_detections.load();
  return 0;
}
