#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <networktables/DoubleArrayTopic.h>
#include <networktables/NetworkTableInstance.h>

#include "control_loop/node.h"

namespace streamer {

// Publishes the densest valid cluster as field coordinates (meters): center,
// major +/− endpoints, minor +/− endpoints, using the tracker's 2-sigma ellipse.
class DensestClusterRioStreamerNode final : public control_loop::INode {
 public:
  DensestClusterRioStreamerNode(std::string_view input_path,
                                std::string_view pose_path,
                                std::string_view networktable_path,
                                nt::NetworkTableInstance instance =
                                    nt::NetworkTableInstance::GetDefault());
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  std::string input_path_, pose_path_, output_path_;
  nt::DoubleArrayPublisher publisher_;
  std::vector<control_loop::MessageDescriptor> dependencies_, publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::mutex mutex_;
};

}  // namespace streamer
