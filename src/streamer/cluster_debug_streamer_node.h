#pragma once

#include <mutex>
#include <nadjieb/mjpeg_streamer.hpp>
#include "control_loop/node.h"

namespace streamer {
// Live field ellipses, exact densest-publisher output, and tracker threshold mask.
class ClusterDebugStreamerNode final : public control_loop::INode {
 public:
  ClusterDebugStreamerNode(std::string clusters, std::string pose,
                           std::string densest, std::string mask, int port = 5805);
  auto CreateCallback() -> std::function<void(const control_loop::Context&)> override;
  auto GetDependencies() const -> const std::vector<control_loop::MessageDescriptor>& override { return dependencies_; }
  auto GetPublications() const -> const std::vector<control_loop::MessageDescriptor>& override { return publications_; }
  void RegisterCallback(const std::function<void(const control_loop::Context&)>& callback) override { callbacks_.push_back(callback); }
 private:
  std::string clusters_, pose_, densest_, mask_;
  std::vector<control_loop::MessageDescriptor> dependencies_, publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::mutex mutex_;
  nadjieb::MJPEGStreamer streamer_;
};
}  // namespace streamer
