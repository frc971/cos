#include "apriltag/gpu_apriltag_detector_node.h"
#include <opencv2/imgproc.hpp>
#include "NvBuffer.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "third_party/971apriltag/apriltag.h"
#include "utils/constants_from_json.h"

namespace apriltag {

GPUApriltagDetectorNode::GPUApriltagDetectorNode(
    uint image_width, uint image_height, const nlohmann::json& intrinsics)
    : camera_matrix_(
          utils::CameraMatrixFromJson<frc::apriltag::CameraMatrix>(intrinsics)),
      distortion_coefficients_(
          utils::DistortionCoefficientsFromJson<frc::apriltag::DistCoeffs>(
              intrinsics)) {
  LOG(INFO) << image_width << " " << image_height;

  apriltag_detector_ = apriltag_detector_create();
  apriltag_detector_add_family_bits(apriltag_detector_, tag36h11_create(), 1);
  apriltag_detector_->nthreads = 6;
  apriltag_detector_->wp = workerpool_create(apriltag_detector_->nthreads);
  apriltag_detector_->qtp.min_white_black_diff = 4;
  apriltag_detector_->debug = false;

  gpu_detector_ = std::make_unique<frc::apriltag::GpuDetector>(
      image_width, image_height, apriltag_detector_,
      utils::CameraMatrixFromJson<frc::apriltag::CameraMatrix>(intrinsics),
      utils::DistortionCoefficientsFromJson<frc::apriltag::DistCoeffs>(
          intrinsics),
      image_format);
}

void GPUApriltagDetectorNode::RegisterCallback(
    const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>&
        callback) {
  callbacks_.push_back(callback);
}

void GPUApriltagDetectorNode::Detect(const camera::DecodedJpegNvBuffer& frame,
                                     double timestamp) {
  cv::Mat gray = NvBufferToGray(frame);
  CHECK(!gray.empty());
  CHECK(gray.channels() == 1);

  auto detections = std::make_shared<std::vector<tag_detection_t>>();
  try {
    absl::Status detection_status = gpu_detector_->Detect(gray.data, nullptr);
    if (!detection_status.ok()) {
      if (restart_detector_on_cuda_error) {
        gpu_detector_ = std::make_unique<frc::apriltag::GpuDetector>(
            gray.cols, gray.rows, apriltag_detector_, camera_matrix_,
            distortion_coefficients_, image_format);
      }
      for (const auto& cb : callbacks_) {
        cb(detections);
      }
      return;
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Returning no detections because of exception: "
                 << e.what();
    for (const auto& cb : callbacks_) {
      cb(detections);
    }
    return;
  }

  const zarray_t* raw_detections = gpu_detector_->Detections();
  detections->reserve(zarray_size(raw_detections));
  for (int i = 0; i < zarray_size(raw_detections); ++i) {
    apriltag_detection_t* gpu_detection;
    zarray_get(raw_detections, i, &gpu_detection);

    tag_detection_t detection;
    detection.tag_id = gpu_detection->id;
    detection.timestamp = timestamp;
    detection.confidence = gpu_detection->decision_margin;
    for (int j = 0; j < 4; ++j) {
      detection.corners[j] =
          cv::Point2d(gpu_detection->p[j][0], gpu_detection->p[j][1]);
    }
    detections->push_back(detection);
  }

  for (const auto& cb : callbacks_) {
    cb(detections);
  }
}

GPUApriltagDetectorNode::~GPUApriltagDetectorNode() {
  if (apriltag_detector_ != nullptr) {
    apriltag_detector_destroy(apriltag_detector_);
  }
}

}  // namespace apriltag
