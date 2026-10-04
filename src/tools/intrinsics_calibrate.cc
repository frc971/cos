#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <nadjieb/mjpeg_streamer.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include "camera/uvc_camera_node.h"
#include "control_loop/context.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "tools/charuco_calibration.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, config_path, "",                 // NOLINT
          "path to the uvc config json file");          // NOLINT
ABSL_FLAG(int, port, 5801, "MJPEG stream port");        // NOLINT
ABSL_FLAG(std::string, stream_path, "/calibration",     // NOLINT
          "MJPEG stream path");                         // NOLINT
ABSL_FLAG(std::string, board_output_path,               // NOLINT
          "calibration_board.png",                      // NOLINT
          "path for the generated ChArUco board PNG");  // NOLINT
ABSL_FLAG(bool, generate_board, false,                  // NOLINT
          "generate the ChArUco board PNG");            // NOLINT
ABSL_FLAG(std::string, intrinsics_output_path,          // NOLINT
          "intrinsics.json",                            // NOLINT
          "path for the generated intrinsics JSON");    // NOLINT
ABSL_FLAG(int, jpeg_quality, 85,                        // NOLINT
          "annotated MJPEG JPEG quality");              // NOLINT

namespace {

using json = nlohmann::json;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

using charuco_calibration::CalibrateCamera;
using charuco_calibration::CreateBoard;
using charuco_calibration::CreateDetector;
using charuco_calibration::DetectCharucoBoard;
using charuco_calibration::DetectionResult;
using charuco_calibration::GenerateBoardImage;
using charuco_calibration::HasEnoughCorners;
using charuco_calibration::IntrinsicsToJson;

auto DrawDetectionResult(cv::Mat& result,
                         const DetectionResult& detection_result) -> void {
  if (!detection_result.marker_ids.empty()) {
    cv::aruco::drawDetectedMarkers(result, detection_result.marker_corners,
                                   detection_result.marker_ids);
  }
  if (!detection_result.charuco_corners.empty()) {
    cv::aruco::drawDetectedCornersCharuco(
        result, detection_result.charuco_corners, detection_result.charuco_ids);
  }
}

auto EncodeJpeg(const cv::Mat& image) -> std::string {
  std::vector<uchar> encoded;
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,
                             absl::GetFlag(FLAGS_jpeg_quality)};
  CHECK(cv::imencode(".jpg", image, encoded, params));
  return std::string(encoded.begin(), encoded.end());
}

auto DecodeJpeg(const camera::JpegBuffer& jpeg_buffer) -> cv::Mat {
  cv::Mat encoded(1, static_cast<int>(jpeg_buffer.size), CV_8UC1,
                  jpeg_buffer.ptr);
  return cv::imdecode(encoded, cv::IMREAD_COLOR);
}

auto WriteIntrinsicsToFile(const cv::Mat& camera_matrix,
                           const cv::Mat& dist_coeffs, const std::string& path)
    -> void {
  std::ofstream intrinsics_file(path);
  CHECK(intrinsics_file.is_open()) << "Failed to open " << path;
  const json intrinsics = IntrinsicsToJson(camera_matrix, dist_coeffs);
  intrinsics_file << intrinsics.dump(4) << '\n';

  std::cout << "Intrinsics:\n" << intrinsics.dump(4) << std::endl;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  control_loop::RioClock::EnableSimulation();
  stop::RegisterHandler();

  CHECK(!absl::GetFlag(FLAGS_config_path).empty())
      << "--config_path is required";
  const int port = absl::GetFlag(FLAGS_port);
  CHECK_GT(port, 0);
  CHECK_LT(port, 65535) << "--port must leave room for the capture stream";

  const cv::aruco::CharucoBoard board = CreateBoard();
  const cv::aruco::CharucoDetector detector = CreateDetector(board);

  if (absl::GetFlag(FLAGS_generate_board)) {
    const cv::Mat board_image = GenerateBoardImage(board);
    CHECK(cv::imwrite(absl::GetFlag(FLAGS_board_output_path), board_image))
        << "Failed to write " << absl::GetFlag(FLAGS_board_output_path);
    std::cout << "Wrote board to " << absl::GetFlag(FLAGS_board_output_path)
              << std::endl;
  }

  camera::UVCCameraConfig config(absl::GetFlag(FLAGS_config_path));
  auto camera_node =
      std::make_shared<camera::UVCCameraNode>("jpeg_stream", config);

  MJPEGStreamer streamer;
  streamer.start(port);
  MJPEGStreamer captured_streamer;
  captured_streamer.start(port + 1);
  cv::Mat captured_canvas;
  std::string captured_jpeg;

  std::mutex detections_mutex;
  std::vector<DetectionResult> detection_results;
  std::atomic<int> pending_captures = 0;
  std::atomic<int> entered_frames = 0;
  std::atomic<bool> calibrate_and_quit = false;
  std::mutex image_size_mutex;
  std::optional<cv::Size> observed_image_size;

  control_loop::ControlLoop control_loop(std::chrono::milliseconds(1));
  control_loop.RegisterDependencyNode(camera_node);
  control_loop.RegisterCallback([&](const control_loop::Context& context) {
    const auto* jpeg_buffer =
        context->GetMessage<camera::JpegBuffer>("jpeg_stream");
    if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr) {
      return;
    }

    cv::Mat frame = DecodeJpeg(*jpeg_buffer);
    if (frame.empty()) {
      LOG(WARNING) << "Failed to decode JPEG frame";
      return;
    }

    {
      std::scoped_lock lock(image_size_mutex);
      if (!observed_image_size.has_value()) {
        observed_image_size = frame.size();
      }
    }

    DetectionResult detection_result = DetectCharucoBoard(frame, detector);
    cv::Mat annotated_frame = frame.clone();
    DrawDetectionResult(annotated_frame, detection_result);
    if (captured_canvas.empty()) {
      captured_canvas = cv::Mat::zeros(frame.size(), frame.type());
      captured_jpeg = EncodeJpeg(captured_canvas);
    }

    size_t captured_count = 0;
    {
      std::scoped_lock lock(detections_mutex);
      captured_count = detection_results.size();
    }
    cv::putText(annotated_frame,
                "entered: " + std::to_string(entered_frames.load()) +
                    "  captured: " + std::to_string(captured_count) +
                    "  corners: " +
                    std::to_string(detection_result.charuco_corners.total()),
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);

    streamer.publish(absl::GetFlag(FLAGS_stream_path),
                     EncodeJpeg(annotated_frame));

    int pending = pending_captures.load();
    while (pending > 0 &&
           !pending_captures.compare_exchange_weak(pending, pending - 1)) {}
    if (pending > 0) {
      const int entered_count = entered_frames.load();
      if (HasEnoughCorners(detection_result)) {
        DrawDetectionResult(captured_canvas, detection_result);
        captured_jpeg = EncodeJpeg(captured_canvas);
        std::scoped_lock lock(detections_mutex);
        detection_results.push_back(std::move(detection_result));
        std::cout << "Captured frame " << detection_results.size() << " of "
                  << entered_count << " entered" << std::endl;
      } else {
        std::cout << "Skipped frame " << entered_count
                  << ": not enough ChArUco corners" << std::endl;
      }
    }
    captured_streamer.publish(absl::GetFlag(FLAGS_stream_path), captured_jpeg);
  });

  camera_node->Start();
  control_loop.Start();

  std::cout << "Streaming annotated frames on port "
            << absl::GetFlag(FLAGS_port) << absl::GetFlag(FLAGS_stream_path)
            << '\n'
            << "Streaming all captured detections on port " << port + 1
            << absl::GetFlag(FLAGS_stream_path) << '\n'
            << "Press Enter to capture a frame, or type q then Enter to "
               "calibrate and quit."
            << std::endl;

  auto process_input = [&](const std::string& input) {
    if (input == "q") {
      calibrate_and_quit.store(true);
      return;
    }
    if (input.empty()) {
      const int entered_count = entered_frames.fetch_add(1) + 1;
      pending_captures.fetch_add(1);
      std::cout << "Frames entered: " << entered_count << std::endl;
      return;
    }
    std::cout << "Unknown command. Press Enter to capture or q to quit."
              << std::endl;
  };

  bool stdin_closed = false;
  std::string input_buffer;
  char input[256];

  while (!stop::stop && !calibrate_and_quit.load()) {
    if (stdin_closed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    pollfd stdin_poll = {
        .fd = STDIN_FILENO,
        .events = POLLIN,
        .revents = 0,
    };
    const int poll_result = poll(&stdin_poll, 1, 50);
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOG(ERROR) << "stdin poll failed with errno " << errno;
      break;
    }
    if (poll_result == 0) {
      continue;
    }
    if ((stdin_poll.revents & POLLIN) == 0) {
      if ((stdin_poll.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        stdin_closed = true;
      }
      continue;
    }

    const ssize_t bytes_read = read(STDIN_FILENO, input, sizeof(input));
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOG(ERROR) << "stdin read failed with errno " << errno;
      break;
    }
    if (bytes_read == 0) {
      stdin_closed = true;
      continue;
    }

    for (ssize_t i = 0; i < bytes_read; ++i) {
      if (input[i] == '\n') {
        if (!input_buffer.empty() && input_buffer.back() == '\r') {
          input_buffer.pop_back();
        }
        process_input(input_buffer);
        input_buffer.clear();
      } else {
        input_buffer.push_back(input[i]);
      }
    }
  }

  control_loop.Stop();
  streamer.stop();
  captured_streamer.stop();

  if (stop::stop && !calibrate_and_quit.load()) {
    return 0;
  }

  cv::Size image_size;
  {
    std::scoped_lock lock(image_size_mutex);
    if (!observed_image_size.has_value()) {
      LOG(ERROR) << "No frames were decoded";
      return 1;
    }
    image_size = *observed_image_size;
  }

  std::vector<DetectionResult> results_snapshot;
  {
    std::scoped_lock lock(detections_mutex);
    results_snapshot = detection_results;
  }

  std::cout << "Calibrating with " << results_snapshot.size()
            << " captured frames" << std::endl;

  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  std::optional<double> reprojection_error = CalibrateCamera(
      results_snapshot, image_size, &camera_matrix, &dist_coeffs);
  if (!reprojection_error.has_value()) {
    LOG(ERROR) << "No usable detections captured";
    return 1;
  }

  std::cout << "Reprojection error: " << *reprojection_error << std::endl;
  WriteIntrinsicsToFile(camera_matrix, dist_coeffs,
                        absl::GetFlag(FLAGS_intrinsics_output_path));
  return 0;
}
