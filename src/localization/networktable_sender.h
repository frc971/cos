#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <wpi/math/geometry/Pose2d.hpp>
#include <wpi/math/geometry/Pose3d.hpp>
#include <wpi/math/geometry/struct/Pose2dStruct.hpp>
#include <wpi/math/geometry/struct/Pose3dStruct.hpp>
#include <wpi/nt/BooleanArrayTopic.hpp>
#include <wpi/nt/BooleanTopic.hpp>
#include <wpi/nt/DoubleArrayTopic.hpp>
#include <wpi/nt/DoubleTopic.hpp>
#include <wpi/nt/IntegerTopic.hpp>
#include <wpi/nt/NetworkTable.hpp>
#include <wpi/nt/NetworkTableInstance.hpp>
#include <wpi/nt/StringTopic.hpp>
#include <wpi/nt/StructTopic.hpp>

#include "localization/position.h"
#include "localization/position_sender.h"

namespace localization {

class NetworkTableSender : public IPositionSender {
 public:
  explicit NetworkTableSender(const std::string& camera_name,
                              bool verbose = false);

  void Send(const position_estimate_t& estimate) override;

  auto MakeDoubleChannel(const std::string& subkey)
      -> std::function<void(double)>;
  auto MakeBoolChannel(const std::string& subkey) -> std::function<void(bool)>;
  auto MakeStringChannel(const std::string& subkey)
      -> std::function<void(std::string)>;

 private:
  wpi::nt::NetworkTableInstance instance_;
  std::shared_ptr<wpi::nt::NetworkTable> table_;

  wpi::nt::StructPublisher<wpi::math::Pose2d> pose_publisher_;
  wpi::nt::StructPublisher<wpi::math::Pose3d> pose3d_publisher_;
  wpi::nt::DoubleArrayPublisher tag_estimation_publisher_;
  wpi::nt::BooleanArrayPublisher tag_ids_publisher_;
  wpi::nt::BooleanArrayPublisher rejected_tag_ids_publisher_;
  wpi::nt::IntegerPublisher num_tags_publisher_;
  wpi::nt::DoublePublisher variance_publisher_;
  wpi::nt::DoublePublisher avg_tag_dist_publisher_;
  wpi::nt::DoublePublisher loss_publisher_;

  std::vector<std::shared_ptr<wpi::nt::DoublePublisher>> double_publishers_;
  std::vector<std::shared_ptr<wpi::nt::BooleanPublisher>> bool_publishers_;
  std::vector<std::shared_ptr<wpi::nt::StringPublisher>> string_publishers_;

  std::mutex mutex_;
  bool verbose_;
};

}  // namespace localization
