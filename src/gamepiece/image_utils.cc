#include "gamepiece/image_utils.h"

#include <limits>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>

namespace gamepiece {

void FindThresholdedPoints(const cv::Mat1b& threshold_mask,
                           std::vector<cv::Point2d>& thresholded_points) {
  std::vector<cv::Point> integer_points;
  cv::findNonZero(threshold_mask, integer_points);
  thresholded_points.assign(integer_points.begin(), integer_points.end());
}

void HSVThreshold(const cv::Mat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  std::vector<cv::Point2d>& thresholded_points,
                  cv::Mat3b& hsv_image, cv::Mat1b& threshold_mask) {
  cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_image, cv::Scalar(minimum_hue, minimum_saturation, 0),
              cv::Scalar(maximum_hue, 255, 255), threshold_mask);
  FindThresholdedPoints(threshold_mask, thresholded_points);
}

void HSVThreshold(const cv::cuda::GpuMat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream) {
  cv::cuda::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV, 0, stream);
  cv::cuda::inRange(hsv_image, cv::Scalar(minimum_hue, minimum_saturation, 0),
                    cv::Scalar(maximum_hue, 255, 255), threshold_mask, stream);
}

void HSVThreshold(const cv::cuda::GpuMat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  std::vector<cv::Point2d>& thresholded_points,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream) {
  HSVThreshold(bgr_image, minimum_hue, maximum_hue, minimum_saturation,
               hsv_image, threshold_mask, stream);
  cv::Mat1b cpu_mask;
  threshold_mask.download(cpu_mask, stream);
  stream.waitForCompletion();
  FindThresholdedPoints(cpu_mask, thresholded_points);
}

auto UndistortedPinholePointOffset(const cv::Point2d& point,
                                   const float world_relative_vertical,
                                   const cv::Matx44f& camera_extrinsics_cv)
    -> std::optional<frc::Translation2d> {
  const cv::Vec4d camera_ray =
      camera_extrinsics_cv * cv::Vec4d{point.x, point.y, 1.0f, 0.0f};

  const float ray_y = camera_ray[1];
  if (ray_y <= std::numeric_limits<float>::epsilon()) {
    return std::nullopt;
  }
  const float scale =
      (world_relative_vertical - camera_extrinsics_cv(1, 3)) / ray_y;
  const cv::Vec4d floor_relative_offset =
      scale * camera_ray;  // relative to the y=0 point below the camera
  return std::make_optional<frc::Translation2d>(
      {units::meter_t{floor_relative_offset[2]},
       units::meter_t{-floor_relative_offset[0]}});
}

}  // namespace gamepiece
