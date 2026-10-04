#include "control_loop/context_handoff_node.h"

#include <cstddef>
#include <mutex>
#include <utility>

#include "absl/log/check.h"

namespace control_loop {

struct ContextHandoffNode::State {
  std::mutex mutex;
  std::vector<std::shared_ptr<IMessage>> messages;
};

ContextHandoffNode::ContextHandoffNode(const std::shared_ptr<INode>& node)
    : state_(std::make_shared<State>()) {
  CHECK(node != nullptr) << "Context handoff source cannot be null";
  publications_ = node->GetPublications();
  state_->messages.resize(publications_.size());
  node->RegisterCallback(
      [weak = std::weak_ptr(state_),
       publications = publications_](const Context& context) -> void {
        if (auto state = weak.lock()) {
          std::scoped_lock lock(state->mutex);
          for (std::size_t i = 0; i < publications.size(); ++i) {
            if (auto message = context->GetSharedMessage<IMessage>(
                    publications[i].GetChannel())) {
              state->messages[i] = std::move(message);
            }
          }
        }
      });
}

auto ContextHandoffNode::CreateCallback()
    -> std::function<void(const Context&)> {
  return [this](const Context& context) -> void {
    {
      std::scoped_lock lock(state_->mutex);
      for (std::size_t i = 0; i < publications_.size(); ++i) {
        if (state_->messages[i]) {
          context->SetMessage(publications_[i].GetChannel(),
                              std::move(state_->messages[i]));
        }
      }
    }
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto ContextHandoffNode::GetDependencies() const
    -> const std::vector<MessageDescriptor>& {
  return dependencies_;
}

auto ContextHandoffNode::GetPublications() const
    -> const std::vector<MessageDescriptor>& {
  return publications_;
}

void ContextHandoffNode::RegisterCallback(
    const std::function<void(const Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace control_loop
