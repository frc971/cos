#include "pathing/navgrid_source_node.h"

namespace pathing {

NavgridSourceNode::NavgridSourceNode(const streamer::NavgridMessage& navgrid)
    : navgrid_(std::make_shared<streamer::NavgridMessage>(navgrid)) {}

auto NavgridSourceNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    context->SetMessage(kNavgrid, navgrid_);
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

auto NavgridSourceNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  static const std::vector<control_loop::MessageDescriptor> dependencies;
  return dependencies;
}

auto NavgridSourceNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  static const std::vector<control_loop::MessageDescriptor> publications{
      {kNavgrid, typeid(streamer::NavgridMessage)}};
  return publications;
}

void NavgridSourceNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace pathing
