#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <networktables/NetworkTableInstance.h>
#include <networktables/StructArrayTopic.h>

#include "control_loop/node.h"
#include "gamepiece/cluster_nt.h"

namespace streamer {

// Joins clusters and pose in the same context, in either arrival order. The
// full NT topic is also a context publication of vector<cluster_nt_t>, allowing
// one send per context even when dependencies finish concurrently.
class ClusterRioStreamerNode final : public control_loop::INode {
 public:
  // NT4 rounds requested periods to 10 ms, then clamps to at least 5 ms.
  // Request below 5 ms to select its 5 ms minimum rather than rounding to 10.
  static constexpr double kTransmitPeriodSeconds = 0.001;

  ClusterRioStreamerNode(
      std::string_view input_path, std::string_view pose_path,
      std::string_view networktable_path,
      nt::NetworkTableInstance instance = nt::NetworkTableInstance::GetDefault());
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
  std::string pose_path_;
  std::string output_path_;
  nt::StructArrayPublisher<gamepiece::cluster_nt_t> publisher_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::mutex mutex_;
};

}  // namespace streamer
