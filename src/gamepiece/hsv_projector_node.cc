#include "gamepiece/hsv_projector_node.h"
#include <frc/geometry/Translation2d.h>
#include <opencv2/calib3d.hpp>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "gamepiece/image_utils.h"

namespace gamepiece {

HSVProjectorNode::HSVProjectorNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics,
    cv::Vec<double, 5> distortion_coeffs)  // NOLINT
    : input_path_(input_path),
      output_path_(output_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          std::vector<frc::Translation2d>>(output_path_)}),
      intrinsics_(intrinsics),
      extrinsics_(extrinsics),
      distortion_coeffs_(distortion_coeffs) {}

auto HSVProjectorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      context->SetMessage(output_path_, nullptr);
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }
    CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
    CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
    const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC3,
                                 frame->destination.channel[0], frame->stride);
    HSVThreshold(image, min_hue, max_hue, min_saturation, thresholded_points_,
                 hsv_image_, threshold_mask_, stream_);
    if (!thresholded_points_.empty()) {
      cv::undistortPoints(thresholded_points_, thresholded_points_, intrinsics_,
                          distortion_coeffs_);
    }
    std::vector<frc::Translation2d> projected_points;
    projected_points.reserve(thresholded_points_.size());
    for (const cv::Point2d& point : thresholded_points_) {
      std::optional<frc::Translation2d> projected_point =
          UndistortedPinholePointOffset(point, 0, extrinsics_);
      if (projected_point.has_value()) {
        projected_points.push_back(
            projected_point.value() +
            frc::Translation2d{units::meter_t{extrinsics_(2, 3)},
                               units::meter_t{-extrinsics_(0, 3)}});
      }
    }

    context->SetMessage(
        output_path_,
        std::make_shared<
            control_loop::ValueMessage<std::vector<frc::Translation2d>>>(
            std::move(projected_points)));
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void HSVProjectorNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto HSVProjectorNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto HSVProjectorNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace gamepiece
