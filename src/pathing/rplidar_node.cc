#include "pathing/rplidar_node.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "sl_lidar_driver.h"

namespace pathing {
RPLidarNode::RPLidarNode(const std::filesystem::path& config_path,
                       std::string_view output_path)
    : output_path_(output_path),
      publications_({{output_path, typeid(RPLidarMessage)}}) {
  std::ifstream file(config_path);
  CHECK(file);

  nlohmann::json data;
  file >> data;

  CHECK(data.at("camera_type") == "rplidar");
  name_ = data.at("name");
  port_ = data.at("port");
  baudrate_ = data.at("baudrate");

  {
    const auto& value = data.at("motor_speed");
    if (!value.is_null()) {
      motor_speed_ = value.get<int>();
    }
  }

  auto channel_result = sl::createSerialPortChannel(port_, baudrate_);
  channel_ = std::unique_ptr<sl::IChannel>(*channel_result);

  auto driver_result = sl::createLidarDriver();
  lidar_ = std::unique_ptr<sl::ILidarDriver>(*driver_result);

  CHECK(SL_IS_OK(lidar_->connect(channel_.get())));

  {
    sl_lidar_response_device_info_t info{};
    CHECK(SL_IS_OK(lidar_->getDeviceInfo(info)));
    LOG(INFO) << "LiDAR replied: model " << static_cast<unsigned>(info.model)
              << ", firmware " << (info.firmware_version >> 8) << '.'
              << (info.firmware_version & 0xff) << ", hardware "
              << static_cast<unsigned>(info.hardware_version) << '\n';
  }

  CHECK(SL_IS_OK(lidar_->startScan(false, true)));
  // Scan startup applies the SDK's default motor speed internally.
  if (motor_speed_.has_value()) {
    CHECK(SL_IS_OK(lidar_->setMotorSpeed(motor_speed_.value())));
  }
}

auto RPLidarNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    Callback(context);
  };
}

void RPLidarNode::Callback(const control_loop::Context& context) {
  auto message = std::make_shared<RPLidarMessage>();
  size_t count = message->nodes.size();
  const auto result = lidar_->grabScanDataHq(message->nodes.data(), count);
  if (SL_IS_OK(result) && count > 0 &&
      SL_IS_OK(lidar_->ascendScanData(message->nodes.data(), count))) {
    message->length = count;
    context->SetMessage(output_path_, message);
  } else {
    context->SetMessage(output_path_, nullptr);
  }
  for (const auto& callback : callbacks_) {
    callback(context);
  }
}

auto RPLidarNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto RPLidarNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void RPLidarNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

RPLidarNode::~RPLidarNode() {
  lidar_->stop();
}

}  // namespace pathing
