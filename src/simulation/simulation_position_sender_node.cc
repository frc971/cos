#include "absl/log/check.h"
#include "absl/log/log.h"

#include "localization/position.h"
#include "simulation/simulation_position_sender_node.h"

#include "nlohmann/json.hpp"

namespace {

auto GetPoseString(const frc::Pose3d& pose, double timestamp = 0)
    -> std::string {
  nlohmann::json json = {{"position",
                          {{"x", pose.X().value()},
                           {"y", pose.Y().value()},
                           {"z", pose.Z().value()}}},
                         {"quaternion",
                          {{"x", pose.Rotation().GetQuaternion().X()},
                           {"y", pose.Rotation().GetQuaternion().Y()},
                           {"z", pose.Rotation().GetQuaternion().Z()},
                           {"w", pose.Rotation().GetQuaternion().W()}}},
                         {"timestamp", timestamp}};
  return json.dump();
}
}  // namespace
namespace simulation {
SimulationPositionSenderNode::SimulationPositionSenderNode(
    std::string_view input_path)
    : input_path_(input_path),
      server_(5805, "127.0.0.1"),
      dependencies_(
          {{input_path_, typeid(localization::PositionEstimateMessage)}}) {
  server_.setOnConnectionCallback(
      [this](
          const std::weak_ptr<ix::WebSocket>& webSocket,
          const std::shared_ptr<ix::ConnectionState>& connectionState) -> void {
        const auto socket = webSocket.lock();
        if (!socket) {
          return;
        }

        socket->setOnMessageCallback(
            [this, webSocket,
             connectionState](const ix::WebSocketMessagePtr& message) -> void {
              const auto socket = webSocket.lock();
              if (!socket)
                return;

              if (message->type == ix::WebSocketMessageType::Open) {
                if (message->openInfo.uri != "/robot") {
                  socket->close(1008, "Only /robot is available");
                  return;
                }

                std::scoped_lock lock(mutex_);
                clients_.insert(socket);
                LOG(INFO) << "Viewer connected from "
                          << connectionState->getRemoteIp();
              } else if (message->type == ix::WebSocketMessageType::Close) {
                std::scoped_lock lock(mutex_);
                clients_.erase(socket);
              }
            });
      });

  const auto [listening, error] = server_.listen();
  CHECK(listening) << "Unable to listen on ws://127.0.0.1:5805"
                   << "/robot: " << error;
  server_.start();
}

auto SimulationPositionSenderNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto SimulationPositionSenderNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto SimulationPositionSenderNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto pose =
        context->GetMessage<localization::PositionEstimateMessage>(input_path_);
    if (pose == nullptr) {
      return;
    }

    std::set<std::shared_ptr<ix::WebSocket>> connectedClients;
    {
      std::scoped_lock lock(mutex_);
      connectedClients = clients_;
    }
    for (const auto& client : connectedClients) {
      client->send(GetPoseString(pose->pose, pose->timestamp));
    }

    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void SimulationPositionSenderNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

SimulationPositionSenderNode::~SimulationPositionSenderNode() {
  server_.stop();
  ix::uninitNetSystem();
}

}  // namespace simulation
