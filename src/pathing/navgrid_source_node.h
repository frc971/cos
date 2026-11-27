#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "control_loop/node.h"
#include "streamer/navgrid_message.h"

namespace pathing {

// Publishes a fixed navgrid snapshot on each control-loop iteration.
class NavgridSourceNode final : public control_loop::INode {
 public:
  static constexpr char kNavgrid[] = "pathing:navgrid";

  explicit NavgridSourceNode(const streamer::NavgridMessage& navgrid);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) override;

 private:
  std::shared_ptr<streamer::NavgridMessage> navgrid_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace pathing
