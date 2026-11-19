#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <print>
#include <thread>
#include "sl_lidar_driver.h"

#ifndef _countof
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0]))
#endif

auto main() -> int {
  const char* port = "/dev/ttyUSB0";
  const int baudrate = 115200;

  auto channelResult = sl::createSerialPortChannel(port, baudrate);
  std::unique_ptr<sl::IChannel> channel(*channelResult);

  auto driverResult = sl::createLidarDriver();
  std::unique_ptr<sl::ILidarDriver> lidar(*driverResult);

  {
    auto res = lidar->connect(channel.get());
    std::cout << SL_IS_OK(res) << '\n';
  }

  sl_lidar_response_device_info_t info{};
  {
    auto res = lidar->getDeviceInfo(info);
    std::cout << SL_IS_OK(res) << '\n';
  }

  std::cout << "LiDAR replied: model " << static_cast<unsigned>(info.model)
            << ", firmware " << (info.firmware_version >> 8) << '.'
            << (info.firmware_version & 0xff) << ", hardware "
            << static_cast<unsigned>(info.hardware_version) << '\n';

  lidar->setMotorSpeed();
  lidar->startScan(false, true);
  for (int i = 0; i < 1000; i++) {
    std::array<sl_lidar_response_measurement_node_hq_t, 8192> nodes{};
    size_t count = _countof(nodes);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    sl_result op_result = lidar->grabScanDataHq(nodes.data(), count);
    if (SL_IS_OK(op_result)) {
      lidar->ascendScanData(nodes.data(), count);
      for (int pos = 0; pos < std::min((int)count, 1); ++pos) {
        std::println(
            "{} theta: {:03.2f} Dist: {:08.2f} Q: {} ",
            (nodes[pos].flag & SL_LIDAR_RESP_HQ_FLAG_SYNCBIT) ? "S " : "  ",
            (nodes[pos].angle_z_q14 * 90.f) / 16384.f,
            nodes[pos].dist_mm_q2 / 4.0f,
            nodes[pos].quality >> SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT);
      }
    }
  }
  lidar->stop();

  return EXIT_SUCCESS;
}
