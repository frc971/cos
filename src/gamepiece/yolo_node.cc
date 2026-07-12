#include "gamepiece/yolo_node.h"

#include <cmath>

#include <wpi/math/geometry/Rotation3d.hpp>
#include <wpi/math/geometry/Transform3d.hpp>
#include <wpi/math/geometry/Translation3d.hpp>
#include <wpi/units/angle.hpp>
#include <wpi/units/length.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace gamepiece {
namespace {
constexpr size_t kMaxDetections = 6;
}  // namespace

YoloNode::YoloNode(const std::string& model_path,
                   const std::vector<std::string>& class_names,
                   const nlohmann::json& intrinsics,
                   const nlohmann::json& extrinsics)
    : yolo_(model_path, false),
      class_names_(class_names),
      cam_cx_(intrinsics.at("cx").get<float>()),
      cam_cy_(intrinsics.at("cy").get<float>()),
      fx_(intrinsics.at("fx").get<float>()),
      fy_(intrinsics.at("fy").get<float>()),
      cam_pitch_(extrinsics.at("rotation_y").get<float>()),
      pinhole_height_(extrinsics.at("translation_z").get<float>()),
      cam_pose_(
          wpi::math::Translation3d{
              wpi::units::meter_t{extrinsics.at("translation_x").get<float>()},
              wpi::units::meter_t{extrinsics.at("translation_y").get<float>()},
              wpi::units::meter_t{extrinsics.at("translation_z").get<float>()}},
          wpi::math::Rotation3d{
              wpi::units::radian_t{extrinsics.at("rotation_x").get<float>()},
              wpi::units::radian_t{extrinsics.at("rotation_y").get<float>()},
              wpi::units::radian_t{extrinsics.at("rotation_z").get<float>()}}) {
  worker_thread_ = std::jthread([this](const std::stop_token& stop_token) {
    while (!stop_token.stop_requested()) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, stop_token] {
          return !tasks_.empty() || stop_token.stop_requested();
        });
        if (tasks_.empty()) {
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      if (!stop_token.stop_requested()) {
        task();
      }
    }
  });
}

YoloNode::~YoloNode() {
  worker_thread_.request_stop();
  cv_.notify_one();
}

void YoloNode::RegisterCallback(
    const std::function<
        void(std::vector<gamepiece_detection_t>, control_loops::MetaDataList,
             std::shared_ptr<control_loops::Context>)>& callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  callbacks_.push_back(callback);
}

void YoloNode::Detect(const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
                      control_loops::MetaDataList metadata,
                      std::shared_ptr<control_loops::Context> ctx) {
  if (metadata.empty()) {
    LOG(WARNING) << "YoloNode received empty metadata";
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push([this, frame, metadata = std::move(metadata),
                 ctx = std::move(ctx)]() mutable {
      RunDetection(frame, std::move(metadata), std::move(ctx));
    });
  }
  cv_.notify_one();
}

void YoloNode::RunDetection(
    const std::shared_ptr<camera::DecodedJpegNvBuffer>& frame,
    control_loops::MetaDataList metadata,
    std::shared_ptr<control_loops::Context> ctx) {
  std::vector<gamepiece_detection_t> detections;
  if (frame != nullptr && frame->buffer != nullptr) {
    cv::Mat mat = NvBufferToMat(frame->buffer);
    std::vector<cv::Rect> bboxes(kMaxDetections);
    std::vector<float> confidences(kMaxDetections);
    std::vector<int> class_ids(kMaxDetections);

    yolo_.Postprocess(mat.rows, mat.cols, yolo_.RunModel(mat), bboxes,
                      confidences, class_ids);
    for (size_t i = 0; i < kMaxDetections; i++) {
      if (bboxes[i].empty()) {
        break;
      }
      detections.push_back(gamepiece_detection_t{
          .pose = ComputePose(bboxes[i]),
          .tracker_id = -1,
          .class_id = class_ids[i],
          .confidence = confidences[i],
      });
    }
  }

  std::vector<decltype(callbacks_)::value_type> callbacks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks = callbacks_;
  }
  for (const auto& callback : callbacks) {
    callback(detections, metadata, ctx);
  }
}

auto YoloNode::NvBufferToMat(NvBuffer* nv_buffer) -> cv::Mat {
  CHECK(nv_buffer != nullptr);
  CHECK_GT(nv_buffer->n_planes, 0U);
  const NvBuffer::NvBufferPlane& y_plane = nv_buffer->planes[0];
  CHECK(y_plane.data != nullptr);

  cv::Mat y(static_cast<int>(y_plane.fmt.height),
            static_cast<int>(y_plane.fmt.width), CV_8UC1, y_plane.data,
            y_plane.fmt.stride);
  return y.clone();
}

auto YoloNode::ComputePose(const cv::Rect& bbox) const -> wpi::math::Pose3d {
  const float center_y = bbox.y + bbox.height / 2.0F;
  const float center_x = bbox.x + bbox.width / 2.0F;
  const float cam_relative_pitch = std::atan2(center_y - cam_cy_, fy_);
  const float phi = cam_relative_pitch + cam_pitch_;
  const float distance = pinhole_height_ / std::sin(phi);
  const float cam_relative_yaw = std::atan2(center_x - cam_cx_, fx_);

  const wpi::math::Transform3d target_pose_cam_relative{
      wpi::math::Translation3d{
          wpi::units::meter_t{distance * std::cos(cam_relative_pitch) *
                              std::cos(cam_relative_yaw)},
          wpi::units::meter_t{distance * std::cos(cam_relative_pitch) *
                              std::sin(cam_relative_yaw)},
          wpi::units::meter_t{distance * -std::sin(cam_relative_pitch)}},
      wpi::math::Rotation3d{wpi::units::radian_t{0.0},
                            wpi::units::radian_t{cam_relative_pitch},
                            wpi::units::radian_t{-cam_relative_yaw}}};

  return cam_pose_.TransformBy(target_pose_cam_relative);
}

}  // namespace gamepiece
