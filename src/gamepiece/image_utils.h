#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <vector>

#include <frc/geometry/Translation2d.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

namespace gamepiece {

// Finds the lowest mask point on columns spaced at 1/256 of the image width
// (at least one pixel) and fills empty columns by linear interpolation.
// Input x values are undistorted;
// x_scale converts them back to approximate image pixels.
template <typename T>
[[nodiscard]] auto LowestPointsByColumn(
    const std::vector<cv::Point_<T>>& points, int image_width,
    double x_scale = 1.0) -> std::vector<cv::Point_<T>> {
  const int column_frequency = std::max(1, image_width / 256);
  if (points.empty())
    return {};
  int min_x = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  for (const auto& point : points) {
    const int x = cvRound(point.x * x_scale);
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
  }
  std::unordered_map<int, T> lowest_by_x;
  for (const auto& point : points) {
    const int x = cvRound(point.x * x_scale);
    auto [it, inserted] = lowest_by_x.emplace(x, point.y);
    if (!inserted)
      it->second = std::max(it->second, point.y);
  }
  std::vector<cv::Point_<T>> sampled;
  for (int x = min_x; x <= max_x; x += column_frequency) {
    const auto it = lowest_by_x.find(x);
    if (it != lowest_by_x.end())
      sampled.emplace_back(static_cast<T>(x / x_scale), it->second);
  }
  if (sampled.empty()) {
    const auto lowest = std::ranges::max_element(points, {}, &cv::Point_<T>::y);
    return {*lowest};
  }
  std::vector<cv::Point_<T>> result;
  for (int x = min_x; x <= max_x; x += column_frequency) {
    const T normalized_x = static_cast<T>(x / x_scale);
    const auto right =
        std::ranges::lower_bound(sampled, normalized_x, {}, &cv::Point_<T>::x);
    if (right == sampled.begin()) {
      result.emplace_back(normalized_x, right->y);
    } else if (right == sampled.end()) {
      result.emplace_back(normalized_x, sampled.back().y);
    } else if (right->x == normalized_x) {
      result.emplace_back(normalized_x, right->y);
    } else {
      const auto left = right - 1;
      const T fraction = (normalized_x - left->x) / (right->x - left->x);
      result.emplace_back(normalized_x,
                          left->y + fraction * (right->y - left->y));
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] auto InterpolatedLowestY(
    const std::vector<cv::Point_<T>>& columns, T x) -> T {
  const auto right =
      std::ranges::lower_bound(columns, x, {}, &cv::Point_<T>::x);
  if (right == columns.begin())
    return right->y;
  if (right == columns.end())
    return columns.back().y;
  if (right->x == x)
    return right->y;
  const auto left = right - 1;
  const T fraction = (x - left->x) / (right->x - left->x);
  return left->y + fraction * (right->y - left->y);
}

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

// Culls normalized rays on or above the horizon. Projection keeps associated
// data together when filtering (e.g. a pixel and its column-bottom point).
template <typename T, typename MatrixT, typename Projection = std::identity>
void CullAboveHorizon(std::vector<T>& points,
                      const cv::Matx<MatrixT, 4, 4>& camera_extrinsics_cv,
                      Projection projection = {}) {
  const cv::Matx44d extrinsics(camera_extrinsics_cv);
  std::erase_if(points, [&](const T& value) {
    const auto& point = std::invoke(projection, value);
    const double ray_y = extrinsics(1, 0) * point.x +
                         extrinsics(1, 1) * point.y + extrinsics(1, 2);
    return !std::isfinite(point.x) || !std::isfinite(point.y) ||
           !(ray_y > std::numeric_limits<double>::epsilon());
  });
}

// Takes focal-length-normalized points, as returned by cv::undistortPoints.
// Callers must cull rays on or above the horizon before projecting.
template <typename T, typename MatrixT>
[[nodiscard]] auto UndistortedPinholePointOffset(
    const cv::Point_<T>& point, float world_relative_vertical,
    const cv::Matx<MatrixT, 4, 4>& camera_extrinsics_cv) -> frc::Translation2d {
  const cv::Vec4d camera_ray =
      cv::Matx44d(camera_extrinsics_cv) * cv::Vec4d{point.x, point.y, 1.0, 0.0};
  const double ray_y = camera_ray[1];
  const double scale =
      (world_relative_vertical - camera_extrinsics_cv(1, 3)) / ray_y;
  const cv::Vec4d floor_relative_offset =
      scale * camera_ray;  // relative to the y=0 point below the camera
  return frc::Translation2d{units::meter_t{floor_relative_offset[2]},
                            units::meter_t{-floor_relative_offset[0]}};
}

// Normalizes pixel coordinates using pinhole intrinsics; does not correct lens
// distortion. For calibrated distortion, use cv::undistortPoints first.
// The normalized ray must be below the horizon.
template <typename T, typename ExtrinsicsT, typename IntrinsicsT>
[[nodiscard]] auto DistortedPinholePointOffset(
    const cv::Point_<T>& point, float world_relative_vertical,
    const cv::Matx<ExtrinsicsT, 4, 4>& camera_extrinsics_cv,
    const cv::Matx<IntrinsicsT, 3, 3>& camera_intrinsics)
    -> frc::Translation2d {
  const cv::Point2d normalized_point{
      (point.x - camera_intrinsics(0, 2)) / camera_intrinsics(0, 0),
      (point.y - camera_intrinsics(1, 2)) / camera_intrinsics(1, 1)};
  return UndistortedPinholePointOffset(
      normalized_point, world_relative_vertical, camera_extrinsics_cv);
}

}  // namespace gamepiece
