#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "control_loop/control_loop.h"
#include "pathing/rplidar_node.h"

ABSL_FLAG(std::string, config, "", "Path to the RPLidar JSON configuration.");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  const auto config = absl::GetFlag(FLAGS_config);
  if (config.empty()) {
    std::cerr << "Required flag: --config=<path to RPLidar JSON configuration>\n";
    return EXIT_FAILURE;
  }

  constexpr auto kOutputPath = "/rplidar";
  auto lidar = std::make_shared<pathing::RPLidarNode>(config, kOutputPath);
  control_loop::ControlLoop loop(std::chrono::milliseconds(100));
  loop.RegisterDependencyNode(lidar);
  loop.RegisterCallback([](const control_loop::Context& context) {
    if (const auto* message =
            context->GetMessage<pathing::RPLidarMessage>(kOutputPath)) {
      std::cout << *message << std::flush;
    }
  });
  loop.Start();

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
