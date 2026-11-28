/**********************************************************************
 Copyright (c) 2020-2024, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include <arpa/inet.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "unitree_lidar_sdk.h"

ABSL_FLAG(std::string, lidar_ip, "192.168.1.62", "Current LiDAR IPv4 address");
ABSL_FLAG(std::string, local_ip, "192.168.1.2", "Current host IPv4 bind address");
ABSL_FLAG(int, lidar_port, 6101, "Current LiDAR UDP port");
ABSL_FLAG(int, local_port, 6201, "Current host UDP bind port");
ABSL_FLAG(std::string, new_lidar_ip, "192.168.123.110", "New LiDAR IPv4 address");
ABSL_FLAG(std::string, new_local_ip, "192.168.123.120", "New UDP destination IPv4 address");
ABSL_FLAG(int, new_lidar_port, 6101, "New LiDAR UDP port");
ABSL_FLAG(int, new_local_port, 6201, "New UDP destination port");
ABSL_FLAG(std::string, gateway, "0.0.0.0", "New LiDAR IPv4 gateway");
ABSL_FLAG(std::string, subnet_mask, "255.255.255.0", "New LiDAR IPv4 subnet mask");

namespace {

bool ParseIp(const char* flag, const std::string& value, void* address) {
  if (inet_pton(AF_INET, value.c_str(), address) == 1) {
    return true;
  }
  std::cerr << "--" << flag << " requires an IPv4 address, got " << value << '\n';
  return false;
}

bool ParsePort(const char* flag, int value, uint16_t& port) {
  if (value > 0 && value <= std::numeric_limits<uint16_t>::max()) {
    port = static_cast<uint16_t>(value);
    return true;
  }
  std::cerr << "--" << flag << " requires a port from 1 to 65535, got "
            << value << '\n';
  return false;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Specify network settings using flags. See --helpfull.\n";
    return 1;
  }

  const std::string lidar_ip = absl::GetFlag(FLAGS_lidar_ip);
  const std::string local_ip = absl::GetFlag(FLAGS_local_ip);
  in_addr address{};
  uint16_t lidar_port;
  uint16_t local_port;
  unilidar_sdk2::LidarIpAddressConfig config{};
  if (!ParseIp("lidar_ip", lidar_ip, &address) ||
      !ParseIp("local_ip", local_ip, &address) ||
      !ParsePort("lidar_port", absl::GetFlag(FLAGS_lidar_port), lidar_port) ||
      !ParsePort("local_port", absl::GetFlag(FLAGS_local_port), local_port) ||
      !ParseIp("new_lidar_ip", absl::GetFlag(FLAGS_new_lidar_ip), config.lidar_ip) ||
      !ParseIp("new_local_ip", absl::GetFlag(FLAGS_new_local_ip), config.user_ip) ||
      !ParsePort("new_lidar_port", absl::GetFlag(FLAGS_new_lidar_port), config.lidar_port) ||
      !ParsePort("new_local_port", absl::GetFlag(FLAGS_new_local_port), config.user_port) ||
      !ParseIp("gateway", absl::GetFlag(FLAGS_gateway), config.gateway) ||
      !ParseIp("subnet_mask", absl::GetFlag(FLAGS_subnet_mask), config.subnet_mask)) {
    return 1;
  }

  auto* reader = unilidar_sdk2::createUnitreeLidarReader();
  if (reader->initializeUDP(lidar_port, lidar_ip, local_port, local_ip)) {
    std::cerr << "Failed to initialize LiDAR UDP.\n";
    return 1;
  }

  sleep(1);
  reader->setLidarIpAddressConfig(config);
  std::cout << "Sent the IP configuration. Reboot the LiDAR to apply it.\n";
  sleep(1);
}
