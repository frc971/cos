#include "gamepiece/lane_density.h"

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/log/log.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/cv_geometry.h"
#include "utils/image_utils.h"

ABSL_FLAG(int, lane_minimum_hue, 18,                       // NOLINT
          "Minimum lane hue in OpenCV HSV units.");       // NOLINT
ABSL_FLAG(int, lane_maximum_hue, 30,                       // NOLINT
          "Maximum lane hue in OpenCV HSV units.");       // NOLINT
ABSL_FLAG(int, lane_minimum_saturation, 150,               // NOLINT
          "Minimum lane saturation in OpenCV HSV units.");  // NOLINT

namespace gamepiece {
namespace {

constexpr double kMinimumCameraDepth = 1e-3;

static inline auto unhomogenize(const cv::Vec3d& v) -> cv::Vec2d {
  if (v[2] == 0) {
    return cv::Vec2d{v[0], v[1]};
  } else {
    return cv::Vec2d{v[0] / v[2], v[1] / v[2]};
  }
}

auto ClipToPositiveCameraDepth(cv::Vec4d& origin, cv::Vec4d& end) -> bool {
  if (origin[2] < kMinimumCameraDepth && end[2] < kMinimumCameraDepth) {
    return false;
  }

  if (origin[2] < kMinimumCameraDepth) {
    const double interpolation =
        (kMinimumCameraDepth - origin[2]) / (end[2] - origin[2]);
    origin += interpolation * (end - origin);
  } else if (end[2] < kMinimumCameraDepth) {
    const double interpolation =
        (kMinimumCameraDepth - end[2]) / (origin[2] - end[2]);
    end += interpolation * (origin - end);
  }
  return true;
}

}  // namespace

LaneDensityTracker::LaneDensityTracker(
    const camera::Intrinsics& intrinsics,
    const camera::Extrinsics& extrinsics) {
  camera_to_image_ = intrinsics.ToMatrix();
  camera_to_robot_cv_ = extrinsics.ToCameraToRobot<cv::Matx44d>().inv();
  utils::ChangeBasis(camera_to_robot_cv_, utils::Basis::kWpiToCv);
  distortion_coeffs_ = intrinsics.ToDistortionCoefficients();
}

auto LaneDensityTracker::GetImageLaneBoundaries(const frc::Pose3d& robot_pose)
    -> std::array<image_lane_segment_t, 2 * num_lanes + 1> {
  const cv::Matx44d robot_to_field_cv = utils::Pose3dToCvMat(robot_pose);
  const cv::Matx44d field_to_camera =
      (robot_to_field_cv * camera_to_robot_cv_).inv();
  const cv::Matx34d camera_to_image = camera_to_image_ * Pi;

  std::array<image_lane_segment_t, 2 * num_lanes + 1> image_relative_lanes{};
  for (size_t i = 0; i < field_relative_lane_boundaries_.size(); ++i) {
    const lane_segment_t& lane = field_relative_lane_boundaries_[i];
    cv::Vec4d camera_relative_origin = field_to_camera * lane.origin;
    cv::Vec4d camera_relative_end = field_to_camera * lane.end;
    if (!ClipToPositiveCameraDepth(camera_relative_origin,
                                  camera_relative_end)) {
      continue;
    }
    const cv::Vec4d camera_relative_midpoint =
        (camera_relative_origin + camera_relative_end) * 0.5;
    image_relative_lanes[i] = {
        .origin = unhomogenize(camera_to_image * camera_relative_origin),
        .end = unhomogenize(camera_to_image * camera_relative_end),
        .midpoint = unhomogenize(camera_to_image * camera_relative_midpoint),
        .valid = true,
    };
  }
  return image_relative_lanes;
}

auto LaneDensityTracker::GetLaneDensities(const cv::Mat& color_image,
                                         const frc::Pose3d& robot_pose)
    -> std::array<float, 2 * num_lanes> {
  std::vector<cv::Point2f> thresholded_points;
  cv::Mat3b hsv_image;
  cv::Mat1b threshold_mask;
  utils::HSVThreshold(color_image, absl::GetFlag(FLAGS_lane_minimum_hue),
                      absl::GetFlag(FLAGS_lane_maximum_hue),
                      absl::GetFlag(FLAGS_lane_minimum_saturation),
                      thresholded_points, hsv_image, threshold_mask);
  cv::undistortImagePoints(thresholded_points, thresholded_points,
                          camera_to_image_, distortion_coeffs_);
  const auto image_relative_lanes = GetImageLaneBoundaries(robot_pose);
  std::array<float, 2 * num_lanes> per_lane_pixel_density{};
  std::optional<size_t> first_valid_lane;
  for (size_t i = 0; i + 1 < image_relative_lanes.size(); ++i) {
    if (image_relative_lanes[i].valid && image_relative_lanes[i + 1].valid) {
      first_valid_lane = i;
      break;
    }
  }
  if (!first_valid_lane.has_value()) {
    return per_lane_pixel_density;
  }

  const cv::Vec2d across_lanes =
      image_relative_lanes[*first_valid_lane + 1].midpoint -
      image_relative_lanes[*first_valid_lane].midpoint;
  const double across_lanes_norm = cv::norm(across_lanes);
  if (across_lanes_norm == std::numeric_limits<double>::epsilon()) {
    LOG(FATAL) << "Impossible: no distance between the lane midpoints";
  }
  const cv::Vec2d across_lanes_uvec = across_lanes / across_lanes_norm;
  const cv::Vec2d along_lanes_uvec{-across_lanes_uvec[1], across_lanes_uvec[0]};
  std::array<line_t, 2 * num_lanes + 1> general_form_along_lane_boundaries{};
  for (size_t i = 0; i < image_relative_lanes.size(); ++i) {
    if (image_relative_lanes[i].valid) {
      const image_lane_segment_t& lane = image_relative_lanes[i];
      general_form_along_lane_boundaries[i] = {
          .a = lane.origin(1) - lane.end(1),
          .b = lane.end(0) - lane.origin(0),
          .c = lane.origin(0) * lane.end(1) - lane.origin(1) * lane.end(0)};
    }
  }
  const image_lane_segment_t& first_lane = image_relative_lanes.front();
  const image_lane_segment_t& last_lane = image_relative_lanes.back();
  const std::pair<line_t, line_t> general_form_across_lane_boundaries = {
      {.a = first_lane.origin(1) - last_lane.origin(1),
       .b = last_lane.origin(0) - first_lane.origin(0),
       .c = first_lane.origin(0) * last_lane.origin(1) -
            first_lane.origin(1) * last_lane.origin(0)},
      {.a = first_lane.end(1) - last_lane.end(1),
       .b = last_lane.end(0) - first_lane.end(0),
       .c = first_lane.end(0) * last_lane.end(1) -
            first_lane.end(1) * last_lane.end(0)}};

  for (const cv::Point2f& image_point : thresholded_points) {
    for (size_t lane_index = 0; lane_index < image_relative_lanes.size() - 1;
         ++lane_index) {
      if (!image_relative_lanes[lane_index].valid ||
          !image_relative_lanes[lane_index + 1].valid) {
        continue;
      }
      const cv::Vec2d vec_point{image_point.x, image_point.y};
      if (general_form_along_lane_boundaries[lane_index].pointInNormalDirection(
              vec_point) == general_form_along_lane_boundaries[lane_index + 1]
                                .pointInNormalDirection(vec_point) ||
          general_form_across_lane_boundaries.first.pointInNormalDirection(
              vec_point) ==
              general_form_across_lane_boundaries.second.pointInNormalDirection(
                  vec_point)) {
        continue;
      }
      per_lane_pixel_density[lane_index]++;
    }
  }

  std::optional<std::pair<cv::Vec2d, cv::Vec2d>> prev_transformed_lane;
  for (size_t i = 0; i < image_relative_lanes.size(); i++) {
    if (!image_relative_lanes[i].valid) {
      if (i != 0) {
        per_lane_pixel_density[i - 1] = 0.0f;
      }
      prev_transformed_lane = std::nullopt;
      continue;
    }
    std::pair<cv::Vec2d, cv::Vec2d> curr_transformed_lane{
        image_relative_lanes[i].origin, image_relative_lanes[i].end};
    if (i != 0) {
      if (!prev_transformed_lane.has_value()) {
        per_lane_pixel_density[i - 1] = 0.0f;
        prev_transformed_lane = curr_transformed_lane;
        continue;
      }

      cv::Vec2d diagonal =
          curr_transformed_lane.second - prev_transformed_lane->first;
      double diag_len = cv::norm(diagonal);
      if (diag_len == 0.0f) {
        per_lane_pixel_density[i - 1] = 0.0f;
        prev_transformed_lane = std::move(curr_transformed_lane);
        continue;
      }
      diagonal /= diag_len;
      cv::Vec2d offset_1 =
          curr_transformed_lane.first - prev_transformed_lane->first;
      cv::Vec2d perpendicular_component_1 =
          offset_1 - diagonal.dot(offset_1) * diagonal;
      cv::Vec2d offset_2 =
          prev_transformed_lane->second - prev_transformed_lane->first;
      cv::Vec2d perpendicular_component_2 =
          offset_2 - diagonal.dot(offset_2) * diagonal;
      double quadrilateral_area =
          0.5 * diag_len * (cv::norm(perpendicular_component_1) +
                           cv::norm(perpendicular_component_2));
      if (quadrilateral_area > 0.0f) {
        per_lane_pixel_density[i - 1] /= quadrilateral_area;
      } else {
        per_lane_pixel_density[i - 1] = 0.0f;
      }
    }
    prev_transformed_lane = curr_transformed_lane;
  }
  return per_lane_pixel_density;
}  // namespace gamepiece
}  // namespace gamepiece
