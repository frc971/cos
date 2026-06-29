#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/apriltag_detector.h"
#include "apriltag/gpu_apriltag_detector_node.h"
#include "apriltag/opencv_apriltag_detector_node.h"
#include "camera/camera_constants.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loops/loop_controller.h"
#include "gamepiece/yolo_node.h"
#include "localization/networktable_sender.h"
#include "localization/sim_sender.h"
#include "localization/unambiguous_solver_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

ABSL_FLAG(bool, sim_sender, false, "Use the in-process simulation sender");
ABSL_FLAG(std::string, sender_name, "localization",
          "NetworkTables PoseEstimate subtable name");
ABSL_FLAG(bool, run_gamepiece, false,
          "Enable the gamepiece YOLO detection loop");
ABSL_FLAG(std::vector<std::string>, gamepiece_class_names,
          std::vector<std::string>({"coral", "algae"}),
          "Ordered class names matching the YOLO model output");

namespace {

auto ReadJsonFile(const std::string& path) -> nlohmann::json {
  std::ifstream file(path);
  CHECK(file.is_open()) << "Failed to open json file: " << path;
  nlohmann::json json;
  file >> json;
  return json;
}

auto SortedUvcCameras() -> std::vector<camera::camera_constant_t> {
  camera::camera_constants_t constants = camera::GetCameraConstants();
  std::vector<camera::camera_constant_t> cameras;
  cameras.reserve(constants.size());
  for (const auto& [name, constant] : constants) {
    if (constant.camera_type == camera::CameraType::UVC) {
      cameras.push_back(constant);
    }
  }
  std::sort(cameras.begin(), cameras.end());
  return cameras;
}

auto MakeDetector(const camera::camera_constant_t& constant)
    -> std::unique_ptr<apriltag::IApriltagDetectorNode> {
  CHECK(constant.intrinsics_path.has_value())
      << "Camera " << constant.name << " is missing intrinsics_path";
  const nlohmann::json intrinsics = ReadJsonFile(*constant.intrinsics_path);

  switch (constant.detector_type) {
    case camera::DetectorType::AUSTIN_GPU:
      CHECK(constant.frame_width.has_value());
      CHECK(constant.frame_height.has_value());
      return std::make_unique<apriltag::GPUApriltagDetectorNode>(
          *constant.frame_width, *constant.frame_height, intrinsics);
    case camera::DetectorType::OPENCV_CPU:
      return std::make_unique<apriltag::OpenCVApriltagDetectorNode>(intrinsics);
    case camera::DetectorType::INVALID:
      LOG(FATAL) << "Camera " << constant.name << " has invalid detector_type";
  }
  LOG(FATAL) << "Unhandled detector_type for camera " << constant.name;
  return nullptr;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::vector<camera::camera_constant_t> camera_constants = SortedUvcCameras();
  CHECK(!camera_constants.empty()) << "No UVC cameras configured";

  std::unique_ptr<localization::IPositionSender> sender;
  if (absl::GetFlag(FLAGS_sim_sender)) {
    sender = std::make_unique<localization::SimSender>();
  } else {
    sender = std::make_unique<localization::NetworkTableSender>(
        absl::GetFlag(FLAGS_sender_name));
  }

  auto solver =
      std::make_unique<localization::UnambiguousSolverNode>(camera_constants);
  solver->RegisterCallback(
      [sender = sender.get()](localization::position_estimate_t estimate,
                              control_loops::MetaDataList metadata,
                              std::shared_ptr<control_loops::Context> ctx) {
        sender->Send(estimate, std::move(metadata), std::move(ctx));
      });
  auto controller = std::make_shared<control_loops::LocalizationLoopController>(
      camera_constants.size());

  std::vector<std::unique_ptr<camera::UVCCameraNode>> cameras;
  std::vector<std::unique_ptr<streamer::JpegBufferStreamerNode>> streamers;
  std::vector<std::unique_ptr<camera::NvjpegDecodeNode>> decoders;
  std::vector<std::unique_ptr<apriltag::IApriltagDetectorNode>> detectors;

  cameras.reserve(camera_constants.size());
  streamers.reserve(camera_constants.size());
  decoders.reserve(camera_constants.size());
  detectors.reserve(camera_constants.size());

  for (size_t i = 0; i < camera_constants.size(); ++i) {
    const camera::camera_constant_t& constant = camera_constants[i];
    const int camera_idx = static_cast<int>(i);

    auto camera = std::make_unique<camera::UVCCameraNode>(
        camera::UVCCameraConfig(constant));
    camera->RegisterCallback(
        [controller, camera_idx](std::shared_ptr<camera::JpegBuffer> frame,
                                 unsigned long timestamp) {
          controller->ReceiveFrame(camera_idx, std::move(frame), timestamp);
        });
    cameras.push_back(std::move(camera));

    if (constant.port.has_value()) {
      const std::string stream_path = "/" + constant.name;
      auto streamer = std::make_unique<streamer::JpegBufferStreamerNode>(
          stream_path, static_cast<int>(*constant.port));
      auto* streamer_ptr = streamer.get();
      controller->RegisterIterationCallback(
          camera_idx,
          [streamer_ptr](const std::shared_ptr<camera::JpegBuffer>& jpeg,
                         control_loops::MetaDataList,
                         std::shared_ptr<control_loops::Context>) {
            if (jpeg != nullptr) {
              streamer_ptr->Stream(jpeg);
            }
          });
      streamers.push_back(std::move(streamer));
    }

    auto decoder = std::make_unique<camera::NvjpegDecodeNode>(constant.name);
    auto detector = MakeDetector(constant);

    auto* detector_ptr = detector.get();
    decoder->RegisterCallback(
        [detector_ptr](
            const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> ctx) {
          detector_ptr->Detect(frame, std::move(metadata), std::move(ctx));
        });

    detector->RegisterCallback(
        [solver = solver.get()](
            std::shared_ptr<std::vector<apriltag::tag_detection_t>> detections,
            control_loops::MetaDataList metadata,
            std::shared_ptr<control_loops::Context> ctx) {
          solver->Accumulate(std::move(detections), std::move(metadata),
                             std::move(ctx));
        });

    auto* decoder_ptr = decoder.get();
    controller->RegisterIterationCallback(
        camera_idx,
        [decoder_ptr](const std::shared_ptr<camera::JpegBuffer>& jpeg,
                      control_loops::MetaDataList metadata,
                      std::shared_ptr<control_loops::Context> ctx) {
          decoder_ptr->Decode(jpeg, std::move(metadata), std::move(ctx));
        });

    decoders.push_back(std::move(decoder));
    detectors.push_back(std::move(detector));
  }

  std::shared_ptr<control_loops::GamepieceLoopController> gamepiece_controller;
  std::vector<std::unique_ptr<gamepiece::YoloNode>> yolo_nodes;
  std::jthread gamepiece_thread;

  if (absl::GetFlag(FLAGS_run_gamepiece)) {
    std::vector<int> gamepiece_camera_indices;
    for (size_t i = 0; i < camera_constants.size(); ++i) {
      if (camera_constants[i].run_gamepiece) {
        gamepiece_camera_indices.push_back(static_cast<int>(i));
      }
    }
    CHECK(!gamepiece_camera_indices.empty())
        << "run_gamepiece is set but no cameras have run_gamepiece=true";

    gamepiece_controller =
        std::make_shared<control_loops::GamepieceLoopController>(
            camera_constants.size());
    const std::vector<std::string> class_names =
        absl::GetFlag(FLAGS_gamepiece_class_names);
    yolo_nodes.reserve(gamepiece_camera_indices.size());

    for (int camera_idx : gamepiece_camera_indices) {
      const auto& constant = camera_constants[camera_idx];
      CHECK(constant.yolo_model_path.has_value())
          << "Camera " << constant.name << " has run_gamepiece=true "
          << "but is missing yolo_model_path";
      CHECK(constant.intrinsics_path.has_value());
      CHECK(constant.extrinsics_path.has_value());

      const nlohmann::json intrinsics = ReadJsonFile(*constant.intrinsics_path);
      const nlohmann::json extrinsics = ReadJsonFile(*constant.extrinsics_path);

      auto yolo_node = std::make_unique<gamepiece::YoloNode>(
          *constant.yolo_model_path, class_names, intrinsics, extrinsics);

      auto* yolo_node_ptr = yolo_node.get();
      auto* gamepiece_ctrl_ptr = gamepiece_controller.get();

      decoders[camera_idx]->RegisterCallback(
          [gamepiece_ctrl_ptr, camera_idx](
              const std::shared_ptr<camera::DecodedJpegNvBuffer>& decoded,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context>) {
            const unsigned long timestamp =
                metadata.empty() ? 0 : metadata.front().timestamp;
            gamepiece_ctrl_ptr->ReceiveFrame(camera_idx, decoded, timestamp);
          });

      gamepiece_controller->RegisterIterationCallback(
          camera_idx,
          [yolo_node_ptr](
              const std::shared_ptr<camera::DecodedJpegNvBuffer>& decoded,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx) {
            yolo_node_ptr->Detect(decoded, std::move(metadata), std::move(ctx));
          });

      yolo_nodes.push_back(std::move(yolo_node));
    }

    gamepiece_thread =
        std::jthread([gamepiece_controller](const std::stop_token&) {
          gamepiece_controller->Run();
        });
  }

  stop::RegisterHandler([controller, gamepiece_controller] {
    controller->RequestStop();
    if (gamepiece_controller != nullptr) {
      gamepiece_controller->RequestStop();
    }
  });

  for (const auto& camera : cameras) {
    camera->Start();
  }

  LOG(INFO) << "Started localization with " << camera_constants.size()
            << " cameras";
  controller->Run();
  if (gamepiece_controller != nullptr) {
    gamepiece_controller->RequestStop();
  }
}
