#include "control_loop/control_loop.h"

#include <chrono>
#include <exception>
#include <unordered_set>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "logging/wpilog_writer.h"

using namespace std::chrono_literals;

namespace control_loop {

ContextInternal::ContextInternal(std::chrono::steady_clock::time_point start,
                                 ControlLoop* control_loop,
                                 std::stop_token stop_token, std::uint64_t id,
                                 std::shared_ptr<logging::WPILogWriter> writer)
    : start(start),
      control_loop(control_loop),
      stop_token(std::move(stop_token)),
      id(id),
      wpilog_writer_(std::move(writer)) {}

ContextInternal::~ContextInternal() {
  if (wpilog_writer_) {
    try {
      wpilog_writer_->Log(*this);
    } catch (const std::exception& error) {
      LOG(ERROR) << "Failed to write context " << id
                 << " to WPILog: " << error.what();
    }
  }
}

ControlLoop::ControlLoop(std::chrono::milliseconds period) : period_(period) {}

void ControlLoop::Start() {
  ValidateNodeGraph();
  if (!wpilog_filename_.empty()) {
    std::vector<MessageDescriptor> log_publications;
    const auto collect = [&log_publications](const auto& nodes) {
      for (const auto& node : nodes) {
        for (const auto& publication : node->GetPublications()) {
          // Image payloads retain their original untyped descriptors and are
          // intentionally excluded. Typed publications have logging metadata.
          if (publication.GetRegistration().has_value()) {
            log_publications.push_back(publication);
          }
        }
      }
    };
    collect(dependency_nodes_);
    collect(nodes_);
    wpilog_writer_ = std::make_shared<logging::WPILogWriter>(wpilog_filename_,
                                                             log_publications);
  }
  RegisterNodeCallbacks();

  contexts_.reserve(max_contexts_);
  for (size_t i = 0; i < max_contexts_; i++) {
    contexts_.push_back(nullptr);
  }

  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    while (!stop_token.stop_requested()) {
      for (size_t i = 0; i < contexts_.size(); i++) {  // NOLINT
        if (contexts_[i] == nullptr || contexts_[i].use_count() == 1) {
          if (contexts_[i] != nullptr && log_latency_) {
            auto now = std::chrono::steady_clock::now();
            auto latency =
                std::chrono::duration<double>(now - contexts_[i]->start);
            if (contexts_[i]->include_in_perfomance_metrics) {
              timestamp_queue_.push(now);
              if (timestamp_queue_.size() > kTimestampQueueMaxSize) {
                timestamp_queue_.pop();
                const std::chrono::duration<double> elapsed =
                    now - timestamp_queue_.front();
                loops_per_second_ =
                    static_cast<double>(timestamp_queue_.size() - 1) /
                    elapsed.count();
                LOG(INFO) << "Average loops per second: " << loops_per_second_;
              }
              LOG(INFO) << "Control loop took " << latency.count() << "s";
            }
          }

          std::stop_source stop_source;
          Context context(new ContextInternal(std::chrono::steady_clock::now(),
                                              this, stop_source.get_token(),
                                              ++loop_count_, wpilog_writer_));
          for (const auto& dependency : dependencies_) {
            dependency(context);
          }

          for (const auto& callback : callbacks_) {
            callback(context);
          }
          contexts_[i] = context;
        }
      }
      std::this_thread::sleep_for(period_);
    }
  });
}

void ControlLoop::Stop() {
  thread_.request_stop();
  if (thread_.joinable()) {
    thread_.join();
  }
  contexts_.clear();
  if (wpilog_writer_) {
    wpilog_writer_->Flush();
    wpilog_writer_.reset();
  }
}

void ControlLoop::RegisterCallback(
    const std::function<void(const Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

void ControlLoop::RegisterDependency(
    const std::function<void(const Context&)>& dependency) {
  dependencies_.emplace_back(dependency);
}

void ControlLoop::RegisterNode(const std::shared_ptr<INode>& node) {
  nodes_.emplace_back(node);
}
void ControlLoop::RegisterDependencyNode(const std::shared_ptr<INode>& node) {
  dependency_nodes_.emplace_back(node);
  dependencies_.emplace_back(node->CreateCallback());
}

void ControlLoop::EnableLatencyLog() {
  log_latency_ = true;
}

void ControlLoop::EnableWPILog(std::string_view filename) {
  wpilog_filename_ = filename;
}

void ControlLoop::ValidateNodeGraph() {
  std::unordered_map<std::string, std::type_index> publishers;
  for (const auto& node : dependency_nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      PCHECK(!publishers.contains(message_descriptor.GetChannel()))
          << "Multiple publishers to the same channel. Channel is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().size() == 1)
          << "Publisher message descriptor has multiple types. Channel is: "
          << message_descriptor.GetChannel();
      publishers.insert({message_descriptor.GetChannel(),
                         *message_descriptor.GetTypes().begin()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      PCHECK(!publishers.contains(message_descriptor.GetChannel()))
          << "Multiple publishers to the same channel. Channel is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().size() == 1)
          << "Publisher message descriptor has multiple types. Channel is: "
          << message_descriptor.GetChannel();
      publishers.insert({message_descriptor.GetChannel(),
                         *message_descriptor.GetTypes().begin()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetDependencies()) {
      PCHECK(publishers.contains(message_descriptor.GetChannel()))
          << "Node channel dependency does has not been registered. Channel "
             "is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().contains(
          publishers.at(message_descriptor.GetChannel())))
          << "Publisher and subscriber channel type does not match. Channel "
             "is: "
          << message_descriptor.GetChannel();
    }
  }
}

void ControlLoop::RegisterNodeCallbacks() {
  std::unordered_map<std::string, INode*> publishers;
  for (const auto& node : dependency_nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      publishers.insert({message_descriptor.GetChannel(), node.get()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      publishers.insert({message_descriptor.GetChannel(), node.get()});
    }
  }

  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetDependencies()) {
      publishers.at(message_descriptor.GetChannel())
          ->RegisterCallback(node->CreateCallback());
    }
  }
}

auto ControlLoop::GetLoopsPerSecond() const -> double {
  return loops_per_second_;
}

void ControlLoop::SetMaxContext(size_t max_contexts) {
  max_contexts_ = max_contexts;
}

}  // namespace control_loop
