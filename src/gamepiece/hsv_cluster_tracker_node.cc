#include "gamepiece/hsv_cluster_tracker_node.h"

#include <memory>

#include <opencv2/core/cuda.hpp>

#include "absl/log/check.h"
#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {

HSVClusterTrackerNode::HSVClusterTrackerNode(
    std::string_view input_path, std::string_view output_path,
    cv::Matx33d intrinsics, cv::Matx44d extrinsics,
    const cv::Vec<double, 5>& distortion_coeffs,
    const float max_ground_distance_m)
    : input_path_(input_path),
      output_path_(output_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<hsv_clusters_t>(
          output_path_)}),
      tracker_(intrinsics, extrinsics, distortion_coeffs,
               max_ground_distance_m) {}

auto HSVClusterTrackerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    std::unique_lock processing_lock(processing_mutex_);
    if (context->Exists(output_path_)) {
      return;
    }
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      tracker_.ProcessFrame(cv::Mat{});
      context->SetMessage(output_path_, nullptr);
    } else {
      CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_BGRI);
      CHECK_GE(frame->stride, static_cast<size_t>(frame->width) * 3U);
      const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC3,
                                   frame->destination.channel[0], frame->stride);
      image.download(frame_);
      tracker_.ProcessFrame(frame_);
      context->SetMessage(
          output_path_,
          std::make_shared<control_loop::ValueMessage<hsv_clusters_t>>(
              *tracker_.GetClusters()));
    }
    processing_lock.unlock();
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void HSVClusterTrackerNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto HSVClusterTrackerNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto HSVClusterTrackerNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace gamepiece
