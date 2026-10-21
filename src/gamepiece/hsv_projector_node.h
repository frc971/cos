#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/cuda.hpp>

#include "control_loop/node.h"

namespace gamepiece {

struct HSVHeatmapConfig {
  double resolution_m = 0.05;
  cv::Point2d min_m{-16.5, -16.5};
  cv::Point2d max_m{16.5, 16.5};
};

// One frame of unsmoothed observations in WPILib meters. Columns increase +X,
// rows increase +Y. Cell (row, col) starts at min_m + resolution_m * (col, row).
// Bounds are half-open; the last row/column may be narrower than resolution_m.
// Weights are sums per cell, so changing resolution preserves total mass.
struct HSVHeatmap {
  double resolution_m;
  cv::Point2d min_m;
  cv::Point2d max_m;
  cv::Mat1d weights;
  cv::Mat1i pixel_counts;
  uint64_t projected_pixels = 0;
  uint64_t outside_pixels = 0;
  double projected_weight = 0;
  double in_bounds_weight = 0;
};

class HSVProjectorNode final : public control_loop::INode {
 public:
  // Without pose_path the grid is robot-relative. With it, consume a Pose2d
  // and transform into that frame before binning (e.g. a field-relative grid).
  HSVProjectorNode(std::string_view input_path, std::string_view output_path,
                   cv::Matx33d intrinsics, cv::Matx44d extrinsics,
                   const cv::Vec<double, 5>& distortion_coeffs,
                   const HSVHeatmapConfig& heatmap_config = {},
                   std::string_view pose_path = "");
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  [[nodiscard]] auto PixelWeight(const cv::Point2d& point, double height) const
      -> double;

  std::string input_path_;
  std::string output_path_;
  std::string pose_path_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex processing_mutex_;
  cv::cuda::Stream stream_;
  cv::cuda::GpuMat hsv_image_, threshold_mask_;
  std::vector<cv::Point2d> thresholded_points_;
  const cv::Matx33d intrinsics_;
  cv::Matx44d extrinsics_cv_;
  const cv::Vec<double, 5> distortion_coeffs_;
  const HSVHeatmapConfig heatmap_config_;
  cv::Size heatmap_size_;
  static constexpr int min_hue = 20, max_hue = 35, min_saturation = 150;
  static constexpr double ball_height_m = 0.150114;  // 5.91 inches
};

}  // namespace gamepiece
