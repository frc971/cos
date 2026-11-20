#include "control_loop/example_node.h"
ExampleNode::ExampleNode() : dependencies_({{}}), publications_({{}}) {}

auto ExampleNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto ExampleNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto ExampleNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void ExampleNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}
