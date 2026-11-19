#pragma once

#include <sl_lidar_driver.h>
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
#include "control_loop/node.h"

namespace pathing {

class RPLidarMessage final : public control_loop::IMessage {
 public:
  RPLidarMessage() = default;
  auto GetType() -> const std::type_info& override {
    return typeid(RPLidarMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this);
  }

  std::array<sl_lidar_response_measurement_node_hq_t, 8192> nodes{};
  // Number of valid samples in nodes.
  std::size_t length = 0;

  friend auto operator<<(std::ostream& os, const RPLidarMessage& message)
      -> std::ostream& {
    const auto& node = message.nodes.front();
    return os << std::format(
               "{} theta: {:03.2f} Dist: {:08.2f} m Q: {} \n",
               (node.flag & SL_LIDAR_RESP_HQ_FLAG_SYNCBIT) ? "S " : "  ",
               (node.angle_z_q14 * 90.f) / 16384.f, node.dist_mm_q2 / 4000.0f,
               node.quality >> SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT);
  }
};

class RPLidarNode final : public control_loop::INode {
 public:
  RPLidarNode(const std::filesystem::path& config_path,
             std::string_view output_path);
  ~RPLidarNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void Callback(const control_loop::Context& context);
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::string output_path_;
  std::string name_;
  std::string port_;
  int baudrate_;
  std::optional<int> motor_speed_;
  std::unique_ptr<sl::IChannel> channel_;
  std::unique_ptr<sl::ILidarDriver> lidar_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace pathing
