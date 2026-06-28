#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loops/context.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

#include "absl/log/globals.h"

ABSL_FLAG(std::string, config_path, "",         // NOLINT
          "path to the uvc config json file");  // NOLINT

ABSL_FLAG(                                                  // NOLINT
    std::optional<std::string>, stream_path, std::nullopt,  // NOLINT
    "Path for the stream. eg url is 10.9.71.101:8080/path. No stream if "  // NOLINT
    "left blank");  // NOLINT

ABSL_FLAG(std::optional<int>, port, std::nullopt,      // NOLINT
          "Streaming port. No stream if left blank");  // NOLINT

ABSL_FLAG(std::optional<std::string>, log_folder, std::nullopt,  // NOLINT
          "Log folder (end with /). No logs if left blank");     // NOLINT

auto CopyPlane(const NvBuffer::NvBufferPlane& plane, int rows, int cols,
               unsigned char* dst) -> void {
  for (int y = 0; y < rows; ++y) {
    std::memcpy(dst + (y * cols), plane.data + (y * plane.fmt.stride), cols);
  }
}

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  stop::RegisterHandler();

  camera::UVCCameraConfig config(absl::GetFlag(FLAGS_config_path));

  auto uvc_camera_node = std::make_unique<camera::UVCCameraNode>(config);
  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>("");

  std::unique_ptr<streamer::JpegBufferStreamerNode> jpeg_buffer_streamer_node;
  if (absl::GetFlag(FLAGS_stream_path).has_value() &&
      absl::GetFlag(FLAGS_port).has_value()) {
    jpeg_buffer_streamer_node =
        std::make_unique<streamer::JpegBufferStreamerNode>(
            absl::GetFlag(FLAGS_stream_path).value(),
            absl::GetFlag(FLAGS_port).value());
    uvc_camera_node->RegisterCallback(
        [streamer = jpeg_buffer_streamer_node.get()](
            const auto& buffer, unsigned long) { streamer->Stream(buffer); });
  }

  uvc_camera_node->RegisterCallback(
      [decoder = nvjpeg_decode_node.get()](const auto& buffer,
                                           unsigned long timestamp) {
        control_loops::MetaDataList metadata{
            control_loops::MetaData{.timestamp = timestamp}};
        decoder->Decode(buffer, metadata,
                        std::make_shared<control_loops::Context>());
      });

  std::atomic<int> frame_index_atomic = 0;
  if (absl::GetFlag(FLAGS_log_folder).has_value()) {

    nvjpeg_decode_node->RegisterCallback(
        [frame_index_atomic = std::ref(frame_index_atomic),
         log_folder = absl::GetFlag(FLAGS_log_folder).value()](
            const std::shared_ptr<camera::DecodedJpegNvBuffer>& buffer,
            control_loops::MetaDataList,
            std::shared_ptr<control_loops::Context>) {
          if (buffer == nullptr) {
            return;
          }
          const int height = buffer->buffer->planes[0].fmt.height;
          const int width = buffer->buffer->planes[0].fmt.width;

          cv::Mat i420(height + (height / 2), width, CV_8UC1);

          auto* y_dst = i420.ptr<unsigned char>(0);
          auto* u_dst = i420.ptr<unsigned char>(height);
          auto* v_dst = i420.ptr<unsigned char>(height + (height / 4));

          int frame_index = frame_index_atomic.get()++;

          CopyPlane(buffer->buffer->planes[0], static_cast<int>(height),
                    static_cast<int>(width), y_dst);
          CopyPlane(buffer->buffer->planes[1], static_cast<int>(height / 2),
                    static_cast<int>(width / 2), u_dst);
          CopyPlane(buffer->buffer->planes[2], static_cast<int>(height / 2),
                    static_cast<int>(width / 2), v_dst);

          cv::Mat out;
          cv::cvtColor(i420, out, cv::COLOR_YUV2BGR_I420);
          cv::imwrite(log_folder + std::to_string(frame_index) + ".png", out);
        });
  }

  uvc_camera_node->Start();

  LOG(INFO) << "Started logging";

  stop::WaitUntilStop();

  uvc_camera_node.reset();
}
