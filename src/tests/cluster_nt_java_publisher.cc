#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numbers>
#include <stop_token>
#include <string>
#include <thread>

#include <networktables/BooleanTopic.h>

#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/position.h"
#include "streamer/cluster_rio_streamer_node.h"

namespace {
using namespace std::chrono_literals;

auto Run(const std::string& host, unsigned int port) -> bool {
  auto instance = nt::NetworkTableInstance::GetDefault();
  instance.SetServer(host, port);
  instance.StartClient4("cos-cluster-java-test");
  auto ready = instance.GetBooleanTopic("/cluster-test/ready").Subscribe(false);
  auto ack = instance.GetBooleanTopic("/cluster-test/ack").Subscribe(false);
  auto done = instance.GetBooleanTopic("/cluster-test/done").Publish();
  streamer::ClusterRioStreamerNode sender("clusters", "pose", "/COS/Clusters",
                                          instance);
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (!instance.IsConnected() || !ready.Get() ||
         !instance.GetServerTimeOffset()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "Java server connection/ready/time-sync timeout\n";
      return false;
    }
    std::this_thread::sleep_for(10ms);
  }
  // Allow subscription propagation before the first sample. No per-frame Flush:
  // this test must exercise the sender's configured network transmit period.
  std::this_thread::sleep_for(500ms);
  const auto callback = sender.CreateCallback();
  auto next = std::chrono::steady_clock::now();
  for (std::uint64_t frame = 0; frame < 1000; ++frame) {
    const auto sequence = frame / 2;  // Identical consecutive observations.
    const std::size_t count = sequence % 40 == 0 ? 0
                            : sequence % 40 == 1 ? 128 : 2;
    gamepiece::hsv_clusters_t clusters(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto& cluster = clusters[index];
      cluster.robot_relative_translation = frc::Translation2d{
          units::meter_t{static_cast<double>(sequence) + 0.25 + index * 0.125},
          units::meter_t{-2.5 - index * 0.25}};
      cluster.ground_covar = {9, 0, 0, 1};
      cluster.density = (sequence % 17) * 0.25 + index + 0.5;
    }
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, frame);
    context->SetMessage(
        "clusters", std::make_shared<
                        control_loop::ValueMessage<gamepiece::hsv_clusters_t>>(
                        std::move(clusters)));
    auto pose = std::make_shared<localization::PositionEstimateMessage>();
    pose->pose = frc::Pose3d{
        units::meter_t{10}, units::meter_t{-3}, units::meter_t{0},
        frc::Rotation3d{units::radian_t{0}, units::radian_t{0},
                        units::radian_t{std::numbers::pi / 2}}};
    context->SetMessage("pose", pose);
    callback(context);
    next += 5ms;
    std::this_thread::sleep_until(next);
  }
  done.Set(true);
  const auto ack_deadline = std::chrono::steady_clock::now() + 15s;
  while (!ack.Get()) {
    if (std::chrono::steady_clock::now() >= ack_deadline) {
      std::cerr << "Java verification acknowledgement timeout\n";
      return false;
    }
    std::this_thread::sleep_for(10ms);
  }
  std::cout << "Java verified all 1000 cluster arrays\n";
  return true;
}
}  // namespace

auto main(int argc, char** argv) -> int {
  if (argc != 3) {
    std::cerr << "Usage: cluster_nt_java_publisher JAVA_SERVER NT4_PORT\n";
    return EXIT_FAILURE;
  }
  return Run(argv[1], static_cast<unsigned int>(std::stoul(argv[2])))
             ? EXIT_SUCCESS : EXIT_FAILURE;
}
