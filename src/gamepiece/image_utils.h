#pragma once

#include <optional>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace gamepiece {

void HSVThreshold(const cv::Mat& bgr_image, int minimum_hue, int maximum_hue,
                  int minimum_saturation,
                  std::vector<cv::Point2d>& thresholded_points,
                  cv::Mat3b& hsv_image, cv::Mat1b& threshold_mask);

// Enqueues conversion and thresholding on stream. Both outputs stay on the GPU.
void HSVThreshold(const cv::cuda::GpuMat& bgr_image, int minimum_hue,
                  int maximum_hue, int minimum_saturation,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null());

// Downloads only the mask and waits for stream before returning CPU points.
void HSVThreshold(const cv::cuda::GpuMat& bgr_image, int minimum_hue,
                  int maximum_hue, int minimum_saturation,
                  std::vector<cv::Point2d>& thresholded_points,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null());

// Takes points in the focal-length-normalized format used by cv::undistortPoints.
// For pixel coordinates, use DistortedPinholePointOffset instead.
[[nodiscard]] auto UndistortedPinholePointOffset(
    const cv::Point2d& point, float world_relative_vertical,
    const cv::Matx44f& camera_extrinsics_cv)
    -> std::optional<frc::Translation2d>;

}  // namespace gamepiece
