#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"

#include "absl/log/initialize.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

#include "absl/log/globals.h"

ABSL_FLAG(int, fps, 0, "FPS");                       // NOLINT
ABSL_FLAG(int, width, 0, "Width");                   // NOLINT
ABSL_FLAG(int, height, 0, "Height");                 // NOLINT
ABSL_FLAG(std::string, serial_id, "", "Serial id");  // NOLINT

ABSL_FLAG(std::string, path, "/stream",                             // NOLINT
          "Path for the stream. eg url is 10.9.71.101:8080/path");  // NOLINT
ABSL_FLAG(int, port, 5801,                                          // NOLINT
          "Streaming port");                                        // NOLINT
ABSL_FLAG(std::string, name, "UVC camera", "Name");                 // NOLINT
ABSL_FLAG(std::optional<int>, max_frame_size, std::nullopt,         // NOLINT
          "Max frame size");                                        // NOLINT
ABSL_FLAG(std::optional<int>, max_payload_size, std::nullopt,       // NOLINT
          "Max payload size");                                      // NOLINT

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

  CHECK(absl::GetFlag(FLAGS_fps) != 0);
  CHECK(absl::GetFlag(FLAGS_width) != 0);
  CHECK(absl::GetFlag(FLAGS_height) != 0);
  CHECK(absl::GetFlag(FLAGS_serial_id) != "");

  camera::UVCCameraConfig config{
      .name = absl::GetFlag(FLAGS_name),
      .serial_id = absl::GetFlag(FLAGS_serial_id),
      .height = absl::GetFlag(FLAGS_height),
      .width = absl::GetFlag(FLAGS_width),
      .fps = absl::GetFlag(FLAGS_fps),
  };
  config.max_payload_size =
      absl::GetFlag(FLAGS_max_payload_size).value_or(config.max_payload_size);
  config.max_frame_size =
      absl::GetFlag(FLAGS_max_frame_size).value_or(config.max_frame_size);

  auto uvc_camera_node = std::make_unique<camera::UVCCameraNode>(config);
  auto jpeg_buffer_streamer_node =
      std::make_unique<streamer::JpegBufferStreamerNode>(
          absl::GetFlag(FLAGS_path), absl::GetFlag(FLAGS_port));

  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>("");

  uvc_camera_node->RegisterCallback(
      [streamer = jpeg_buffer_streamer_node.get()](const auto& buffer) {
        streamer->Stream(buffer);
      });

  uvc_camera_node->RegisterCallback(
      [decoder = nvjpeg_decode_node.get()](const auto& buffer) {

      });

  // const std::function<void(std::shared_ptr<JpegBuffer>)>& callback) {
  // uvc_camera_node->RegisterCallback([](const std::shared_ptr<camera::JpegBuffer>& buffer) {
  //         buff
  //         });

  uvc_camera_node->Start();

  LOG(INFO) << "Started streaming";

  stop::WaitUntilStop();

  uvc_camera_node.reset();
}
