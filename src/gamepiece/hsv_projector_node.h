#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/cuda.hpp>

#include "control_loop/node.h"

namespace gamepiece {

class HSVProjectorNode final : public control_loop::INode {
 public:
  HSVProjectorNode(std::string_view input_path, std::string_view output_path,
                   cv::Matx33d intrinsics, cv::Matx44d extrinsics,
                   cv::Vec4d distortion_coeffs);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  std::string input_path_;
  std::string output_path_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  cv::cuda::Stream stream_;
  cv::cuda::GpuMat hsv_image_, threshold_mask_;
  std::vector<cv::Point2d> thresholded_points_;
  const cv::Matx33d intrinsics_;
  const cv::Matx44d extrinsics_;
  const cv::Vec4d distortion_coeffs_;
  static constexpr int min_hue = 30, max_hue = 60, min_saturation = 150;
};

}  // namespace gamepiece
