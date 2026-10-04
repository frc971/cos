#include "gamepiece/gamepiece_control_loop.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "camera/nvjpeg_decode_node.h"

namespace gamepiece {

struct GamepieceControlLoop::DecodedFrameState {
  std::mutex mutex;
  std::condition_variable frame_available;
  std::vector<std::shared_ptr<camera::DecodedJpegBuffer>> decoded_buffers;
  std::vector<double> timestamps;
};

GamepieceControlLoop::GamepieceControlLoop(std::chrono::milliseconds period)
    : decoded_frame_state_(std::make_shared<DecodedFrameState>()),
      period_(period) {
  CHECK_GT(period_.count(), 0);
}

GamepieceControlLoop::~GamepieceControlLoop() {
  Stop();
}

void GamepieceControlLoop::RegisterDecodedFrameSource(
    const std::shared_ptr<control_loop::INode>& decoder,
    std::string_view decoded_channel) {
  CHECK(!started_) << "Cannot register a decoded source after starting";
  CHECK(decoder != nullptr) << "Decoded frame source cannot be null";
  CHECK(decoded_channels_.insert(std::string(decoded_channel)).second)
      << "Decoded frame channel was registered more than once: "
      << decoded_channel;

  const size_t source_index = decoded_channels_in_order_.size();
  const std::string channel(decoded_channel);
  decoded_channels_in_order_.push_back(channel);
  decoded_frame_callbacks_.emplace_back();
  {
    std::lock_guard lock(decoded_frame_state_->mutex);
    decoded_frame_state_->decoded_buffers.push_back(nullptr);
    decoded_frame_state_->timestamps.push_back(0.0);
  }

  std::weak_ptr<DecodedFrameState> weak_state = decoded_frame_state_;
  decoder->RegisterCallback(
      [weak_state, source_index,
       channel](const control_loop::Context& localization_context) {
        auto frame =
            localization_context->GetSharedMessage<camera::DecodedJpegBuffer>(
                channel);
        auto state = weak_state.lock();
        if (frame == nullptr || state == nullptr) {
          return;
        }

        {
          std::lock_guard lock(state->mutex);
          state->timestamps[source_index] = frame->timestamp;
          state->decoded_buffers[source_index] = std::move(frame);
        }
        state->frame_available.notify_one();
      });
}

void GamepieceControlLoop::RegisterNode(
    const std::shared_ptr<control_loop::INode>& node) {
  CHECK(!started_) << "Cannot register a node after starting";
  CHECK(node != nullptr) << "Gamepiece node cannot be null";
  nodes_.push_back(node);
}

void GamepieceControlLoop::Start() {
  CHECK(!started_) << "Gamepiece control loop can only be started once";
  CHECK(!decoded_channels_in_order_.empty())
      << "Gamepiece control loop requires a decoded frame source";
  ValidateNodeGraph();
  RegisterNodeCallbacks();
  started_ = true;
  thread_ = std::jthread(
      [this](std::stop_token stop_token) { Run(std::move(stop_token)); });
}

void GamepieceControlLoop::Stop() {
  if (!started_) {
    return;
  }
  thread_.request_stop();
  decoded_frame_state_->frame_available.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void GamepieceControlLoop::ValidateNodeGraph() {
  std::unordered_map<std::string, std::type_index> publishers;
  for (const std::string& channel : decoded_channels_in_order_) {
    publishers.emplace(channel, typeid(camera::DecodedJpegBuffer));
  }

  for (const auto& node : nodes_) {
    for (const auto& publication : node->GetPublications()) {
      CHECK(!publishers.contains(publication.GetChannel()))
          << "Multiple publishers to the same channel. Channel is: "
          << publication.GetChannel();
      CHECK_EQ(publication.GetTypes().size(), 1U)
          << "Publisher message descriptor has multiple types. Channel is: "
          << publication.GetChannel();
      publishers.emplace(publication.GetChannel(),
                         *publication.GetTypes().begin());
    }
  }

  for (const auto& node : nodes_) {
    for (const auto& dependency : node->GetDependencies()) {
      CHECK(publishers.contains(dependency.GetChannel()))
          << "Node channel dependency has not been registered. Channel is: "
          << dependency.GetChannel();
      CHECK(dependency.GetTypes().contains(
          publishers.at(dependency.GetChannel())))
          << "Publisher and subscriber channel type does not match. Channel "
             "is: "
          << dependency.GetChannel();
    }
  }
}

void GamepieceControlLoop::RegisterNodeCallbacks() {
  std::unordered_map<std::string,
                     std::function<void(const std::function<void(
                                            const control_loop::Context&)>&)>>
      callback_registrars;

  for (size_t i = 0; i < decoded_channels_in_order_.size(); ++i) {
    callback_registrars.emplace(
        decoded_channels_in_order_[i], [this, i](const auto& callback) {
          decoded_frame_callbacks_[i].push_back(callback);
        });
  }
  for (const auto& node : nodes_) {
    for (const auto& publication : node->GetPublications()) {
      callback_registrars.emplace(
          publication.GetChannel(),
          [node](const auto& callback) { node->RegisterCallback(callback); });
    }
  }

  for (const auto& node : nodes_) {
    for (const auto& dependency : node->GetDependencies()) {
      callback_registrars.at(dependency.GetChannel())(node->CreateCallback());
    }
  }
}

void GamepieceControlLoop::Run(std::stop_token stop_token) {
  auto next_tick = std::chrono::steady_clock::now() + period_;
  std::uint64_t context_id = 0;
  while (!stop_token.stop_requested()) {
    std::vector<std::shared_ptr<camera::DecodedJpegBuffer>> decoded_buffers;
    std::vector<double> timestamps;
    {
      std::unique_lock lock(decoded_frame_state_->mutex);
      decoded_frame_state_->frame_available.wait_until(
          lock, next_tick, [&] { return stop_token.stop_requested(); });
      if (stop_token.stop_requested()) {
        return;
      }
      next_tick += period_;
      if (!std::ranges::any_of(
              decoded_frame_state_->decoded_buffers,
              [](const auto& buffer) { return buffer != nullptr; })) {
        continue;
      }
      decoded_buffers = decoded_frame_state_->decoded_buffers;
      timestamps = decoded_frame_state_->timestamps;
      std::ranges::fill(decoded_frame_state_->decoded_buffers, nullptr);
    }

    auto context = control_loop::Context(new control_loop::ContextInternal(
        std::chrono::steady_clock::now(), nullptr, stop_token, ++context_id));

    for (size_t i = 0; i < decoded_buffers.size(); ++i) {
      if (decoded_buffers[i] == nullptr) {
        continue;
      }
      CHECK_EQ(decoded_buffers[i]->timestamp, timestamps[i]);
      context->SetMessage(decoded_channels_in_order_[i], decoded_buffers[i]);
      for (const auto& callback : decoded_frame_callbacks_[i]) {
        callback(context);
      }
    }

    while (context.use_count() > 1 && !stop_token.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto finished = std::chrono::steady_clock::now();
    if (finished > next_tick) {
      LOG(WARNING) << "Gamepiece command loop overran: elapsed since tick "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          finished - (next_tick - period_))
                          .count()
                   << " ms (period " << period_.count() << " ms)";
      next_tick += ((finished - next_tick) / period_ + 1) * period_;
    }
  }
}

}  // namespace gamepiece
