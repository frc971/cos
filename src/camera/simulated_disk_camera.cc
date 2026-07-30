#include "camera/simulated_disk_camera.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

#include "absl/log/check.h"

namespace camera {
namespace {

auto ReadBytes(const std::filesystem::path& path)
    -> std::vector<unsigned char> {
  std::ifstream input(path, std::ios::binary);
  CHECK(input.is_open()) << "Failed to open simulated camera frame: " << path;
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

auto ReadFrames(const std::vector<std::filesystem::path>& paths)
    -> std::vector<std::vector<unsigned char>> {
  std::vector<std::vector<unsigned char>> frames;
  frames.reserve(paths.size());
  for (const auto& path : paths) {
    frames.push_back(ReadBytes(path));
  }
  return frames;
}

}  // namespace

SimulatedDiskCamera::SimulatedDiskCamera(
    std::vector<std::filesystem::path> frame_paths,
    unsigned long initial_timestamp,
    std::chrono::microseconds simulation_time_between_frames,
    std::chrono::microseconds playback_time_between_frames)
    : SimulatedDiskCamera(ReadFrames(frame_paths), initial_timestamp,
                          simulation_time_between_frames,
                          playback_time_between_frames) {}

SimulatedDiskCamera::SimulatedDiskCamera(
    std::vector<std::vector<unsigned char>> mjpeg_payloads,
    unsigned long initial_timestamp,
    std::chrono::microseconds simulation_time_between_frames,
    std::chrono::microseconds playback_time_between_frames)
    : mjpeg_payloads_(std::move(mjpeg_payloads)),
      initial_timestamp_(initial_timestamp),
      simulation_interval_(simulation_time_between_frames),
      playback_interval_(playback_time_between_frames ==
                                 std::chrono::microseconds::zero()
                             ? simulation_time_between_frames
                             : playback_time_between_frames) {
  CHECK(!mjpeg_payloads_.empty());
  CHECK(simulation_interval_.count() > 0);
  CHECK(playback_interval_.count() > 0);
  for (const auto& payload : mjpeg_payloads_) {
    CHECK(!payload.empty());
  }
}

SimulatedDiskCamera::~SimulatedDiskCamera() = default;

void SimulatedDiskCamera::RegisterCallback(const CameraCallback& callback) {
  CHECK(!replay_thread_.joinable())
      << "Callbacks must be registered before the camera starts";
  callbacks_.push_back(callback);
}

void SimulatedDiskCamera::Start() {
  CHECK(!replay_thread_.joinable()) << "SimulatedDiskCamera already started";
  replay_thread_ = std::jthread(
      [this](const std::stop_token& stop_token) { Replay(stop_token); });
}

void SimulatedDiskCamera::Replay(const std::stop_token& stop_token) {
  auto next_frame_time = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mjpeg_payloads_.size() && !stop_token.stop_requested();
       ++i) {
    if (i != 0) {
      next_frame_time += playback_interval_;
      std::this_thread::sleep_until(next_frame_time);
      if (stop_token.stop_requested()) {
        return;
      }
    }

    CHECK(i <= (std::numeric_limits<unsigned long>::max() -
                initial_timestamp_) /
                   static_cast<unsigned long>(simulation_interval_.count()))
        << "Simulated camera timestamp overflow";
    const unsigned long timestamp =
        initial_timestamp_ +
        static_cast<unsigned long>(i) *
            static_cast<unsigned long>(simulation_interval_.count());
    const auto& payload = mjpeg_payloads_[i];
    auto buffer = std::make_shared<JpegBuffer>(payload.size());
    std::memcpy(buffer->ptr(), payload.data(), payload.size());
    for (const auto& callback : callbacks_) {
      callback(buffer, timestamp);
    }
  }
}

}  // namespace camera
