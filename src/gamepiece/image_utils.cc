#include "gamepiece/image_utils.h"

#include <limits>

#include <absl/log/log.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace utils {

void HSVThreshold(const cv::Mat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  std::vector<cv::Point2f>& thresholded_points,
                  cv::Mat3b& hsv_image, cv::Mat1b& threshold_mask) {
  cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
  cv::inRange(hsv_image, cv::Scalar(minimum_hue, minimum_saturation, 0),
              cv::Scalar(maximum_hue, 255, 255), threshold_mask);
  cv::findNonZero(threshold_mask, thresholded_points);
}

auto DistortedPointOffset(const cv::Point2f& point,
                          const float world_relative_vertical,
                          const cv::Matx44f& camera_extrinsics_cv,
                          const cv::Matx33f& camera_intrinsics)
    -> std::optional<frc::Translation2d> {
  cv::Point2f normalized_point{
      (point.x - camera_extrinsics_cv(0, 2)) / camera_intrinsics(0, 0),
      (point.y - camera_extrinsics_cv(1, 2)) / camera_intrinsics(1, 1)};
  return UndistortedPinholePointOffset(point, world_relative_vertical,
                                       camera_extrinsics_cv);
}

auto UndistortedPinholePointOffset(const cv::Point2f& point,
                                   const float world_relative_vertical,
                                   const cv::Matx44f& camera_extrinsics_cv)
    -> std::optional<frc::Translation2d> {
  const cv::Vec4f camera_ray =
      camera_extrinsics_cv * cv::Vec4f{point.x, point.y, 1.0f, 0.0f};

  const float ray_y = camera_ray[1];
  if (ray_y <= std::numeric_limits<float>::epsilon()) {
    return std::nullopt;
  }
  const float scale =
      (world_relative_vertical - camera_extrinsics_cv(1, 3)) / ray_y;
  const cv::Vec4f floor_relative_offset =
      scale * camera_ray;  // relative to the y=0 point below the camera
  return std::make_optional<frc::Translation2d>(
      {units::meter_t{floor_relative_offset[2]},
       units::meter_t{-floor_relative_offset[0]}});
}

}  // namespace utils
