#pragma once

#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

#include "camera/camera.h"

namespace camera {
    
class SimulatedDiskCamera : public ICamera {
 public:
  SimulatedDiskCamera(
      std::vector<std::filesystem::path> frame_paths,
      unsigned long initial_timestamp,
      std::chrono::microseconds simulation_time_between_frames,
      std::chrono::microseconds playback_time_between_frames =
          std::chrono::microseconds::zero());
  SimulatedDiskCamera(
      std::vector<std::vector<unsigned char>> mjpeg_payloads,
      unsigned long initial_timestamp,
      std::chrono::microseconds simulation_time_between_frames,
      std::chrono::microseconds playback_time_between_frames =
          std::chrono::microseconds::zero());
  ~SimulatedDiskCamera() override;

  SimulatedDiskCamera(const SimulatedDiskCamera&) = delete;
  SimulatedDiskCamera& operator=(const SimulatedDiskCamera&) = delete;
  SimulatedDiskCamera(SimulatedDiskCamera&&) = delete;
  SimulatedDiskCamera& operator=(SimulatedDiskCamera&&) = delete;

  void RegisterCallback(const CameraCallback& callback) override;
  void Start() override;

 private:
  void Replay(const std::stop_token& stop_token);

  std::vector<std::vector<unsigned char>> mjpeg_payloads_;
  unsigned long initial_timestamp_;
  std::chrono::microseconds simulation_interval_;
  std::chrono::microseconds playback_interval_;
  std::vector<CameraCallback> callbacks_;
  std::jthread replay_thread_;
};

}  // namespace camera
