#pragma once

#include <functional>
#include <networktables/DoubleArrayTopic.h>
#include <networktables/NetworkTableInstance.h>
#include <string>
#include <string_view>
#include <vector>

#include "control_loop/node.h"

namespace streamer {

class BallVelocityGoalStreamerNode final : public control_loop::INode {
 public:
  BallVelocityGoalStreamerNode(std::string_view input_path,
                               std::string_view networktable_path);

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
  nt::NetworkTableInstance instance_;
  nt::DoubleArrayPublisher publisher_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

}  // namespace streamer
