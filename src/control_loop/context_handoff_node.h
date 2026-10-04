#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <vector>

#include "control_loop/node.h"

namespace control_loop {

// Register this node as a dependency in the receiving loop. It retains only
// the latest non-null message per source publication, without retaining source
// contexts. Each receiving callback consumes the pending messages once.
// Register downstream callbacks before starting the receiving loop.
class ContextHandoffNode final : public INode {
 public:
  explicit ContextHandoffNode(const std::shared_ptr<INode>& node);

  ContextHandoffNode(const ContextHandoffNode&) = delete;
  auto operator=(const ContextHandoffNode&) -> ContextHandoffNode& = delete;

  auto CreateCallback() -> std::function<void(const Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<MessageDescriptor>& override;
  void RegisterCallback(
      const std::function<void(const Context&)>& callback) override;

 private:
  struct State;
  std::shared_ptr<State> state_;
  std::vector<MessageDescriptor> publications_;
  std::vector<MessageDescriptor> dependencies_;
  std::vector<std::function<void(const Context&)>> callbacks_;
};

}  // namespace control_loop
