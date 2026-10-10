#pragma once

#include <optional>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <opencv2/core.hpp>

namespace utils {

void HSVThreshold(const cv::Mat& bgr_image, int minimum_hue, int maximum_hue,
                  int minimum_saturation,
                  std::vector<cv::Point2f>& thresholded_points,
                  cv::Mat3b& hsv_image, cv::Mat1b& threshold_mask);

[[nodiscard]] auto DistortedPinholePointOffset(
    const cv::Point2f& point, float world_relative_vertical,
    const cv::Matx44f& camera_extrinsics_cv,
    const cv::Matx33f& camera_intrinsics) -> std::optional<frc::Translation2d>;

// Takes in points in the focal-length-normalized format used by cv::undistortPoints. If points aren't undistorted, use DistortedPinholePointOffset instead
[[nodiscard]] auto UndistortedPinholePointOffset(
    const cv::Point2f& point, float world_relative_vertical,
    const cv::Matx44f& camera_extrinsics_cv)
    -> std::optional<frc::Translation2d>;

}  // namespace utils
