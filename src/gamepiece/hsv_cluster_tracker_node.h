#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "control_loop/node.h"
#include "gamepiece/hsv_cluster_tracker.h"

namespace gamepiece {

using hsv_clusters_t = std::vector<kmeans_cluster_t>;

class HSVClusterTrackerNode final : public control_loop::INode {
 public:
  HSVClusterTrackerNode(std::string_view input_path,
                        std::string_view output_path, cv::Matx33d intrinsics,
                        cv::Matx44d extrinsics,
                        const cv::Vec<double, 5>& distortion_coeffs,
                        float max_ground_distance_m = 20.0f);
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
  std::mutex processing_mutex_;
  HSVClusterTracker tracker_;
  cv::Mat frame_;
};

}  // namespace gamepiece
