#pragma once

#include <limits>
#include <optional>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

namespace gamepiece {

template <typename T>
void FindThresholdedPoints(const cv::Mat1b& threshold_mask,
                           std::vector<cv::Point_<T>>& thresholded_points) {
  std::vector<cv::Point> integer_points;
  cv::findNonZero(threshold_mask, integer_points);
  thresholded_points.assign(integer_points.begin(), integer_points.end());
}

template <typename T>
void HSVThreshold(const cv::Mat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  std::vector<cv::Point_<T>>& thresholded_points,
                  cv::Mat3b& hsv_image, cv::Mat1b& threshold_mask) {
  cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_image, cv::Scalar(minimum_hue, minimum_saturation, 0),
              cv::Scalar(maximum_hue, 255, 255), threshold_mask);
  FindThresholdedPoints(threshold_mask, thresholded_points);
}

// Enqueues conversion and thresholding on stream. Both outputs stay on the GPU.
void HSVThreshold(const cv::cuda::GpuMat& bgr_image, int minimum_hue,
                  int maximum_hue, int minimum_saturation,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null());

// Downloads only the mask and waits for stream before returning CPU points.
template <typename T>
void HSVThreshold(const cv::cuda::GpuMat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  std::vector<cv::Point_<T>>& thresholded_points,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null()) {
  HSVThreshold(bgr_image, minimum_hue, maximum_hue, minimum_saturation,
               hsv_image, threshold_mask, stream);
  cv::Mat1b cpu_mask;
  threshold_mask.download(cpu_mask, stream);
  stream.waitForCompletion();
  FindThresholdedPoints(cpu_mask, thresholded_points);
}

// Takes focal-length-normalized points, as returned by cv::undistortPoints.
template <typename T, typename MatrixT>
[[nodiscard]] auto UndistortedPinholePointOffset(
    const cv::Point_<T>& point, float world_relative_vertical,
    const cv::Matx<MatrixT, 4, 4>& camera_extrinsics_cv)
    -> std::optional<frc::Translation2d> {
  const cv::Vec4d camera_ray =
      cv::Matx44d(camera_extrinsics_cv) * cv::Vec4d{point.x, point.y, 1.0, 0.0};
  const double ray_y = camera_ray[1];
  if (ray_y <= std::numeric_limits<double>::epsilon()) {
    return std::nullopt;
  }
  const double scale =
      (world_relative_vertical - camera_extrinsics_cv(1, 3)) / ray_y;
  const cv::Vec4d floor_relative_offset =
      scale * camera_ray;  // relative to the y=0 point below the camera
  return frc::Translation2d{units::meter_t{floor_relative_offset[2]},
                            units::meter_t{-floor_relative_offset[0]}};
}

// Normalizes pixel coordinates using pinhole intrinsics; does not correct lens
// distortion. For calibrated distortion, use cv::undistortPoints first.
template <typename T, typename ExtrinsicsT, typename IntrinsicsT>
[[nodiscard]] auto DistortedPinholePointOffset(
    const cv::Point_<T>& point, float world_relative_vertical,
    const cv::Matx<ExtrinsicsT, 4, 4>& camera_extrinsics_cv,
    const cv::Matx<IntrinsicsT, 3, 3>& camera_intrinsics)
    -> std::optional<frc::Translation2d> {
  const cv::Point2d normalized_point{
      (point.x - camera_intrinsics(0, 2)) / camera_intrinsics(0, 0),
      (point.y - camera_intrinsics(1, 2)) / camera_intrinsics(1, 1)};
  return UndistortedPinholePointOffset(
      normalized_point, world_relative_vertical, camera_extrinsics_cv);
}

}  // namespace gamepiece
