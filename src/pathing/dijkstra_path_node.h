#pragma once

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <frc/geometry/Pose2d.h>
#include <units/angular_velocity.h>
#include <units/velocity.h>

#include "control_loop/node.h"
#include "streamer/navgrid_streamer_node.h"

namespace pathing {

class PoseMessage final : public control_loop::IMessage {
 public:
  explicit PoseMessage(frc::Pose2d pose) : goal_position(pose) {}
  auto GetType() -> const std::type_info& override {
    return typeid(PoseMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  frc::Pose2d goal_position;
};

enum class PathStatus { FollowingPath, ReachedGoal, InvalidInput, NoPath };

class VelocityGoalMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(VelocityGoalMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
  // Field-relative axes: +x/+y match the input poses and navgrid.
  units::meters_per_second_t vx{0};
  units::meters_per_second_t vy{0};
  units::radians_per_second_t omega{0};
  PathStatus status = PathStatus::InvalidInput;
};

struct MotionSettings {
  units::meters_per_second_t max_translation{1};
  units::radians_per_second_t max_rotation{2};
  double translation_gain = 1;  // 1/s
  double heading_gain = 2;      // 1/s
  units::meter_t position_tolerance{0.05};
  units::radian_t heading_tolerance{0.05235987755982989};  // 3 degrees
  // Extra path cost in meters per radian of travel-direction change.
  // Zero disables the penalty. The first move has no incoming direction.
  double turn_penalty = 0.3;
};

class DijkstraPathNode final : public control_loop::INode {
 public:
  DijkstraPathNode(std::string_view pose_path, std::string_view goal_path,
                   std::string_view output_path,
                   const std::filesystem::path& navgrid_path =
                       "/root/constants/assets/navgrid.json",
                   MotionSettings settings = {});
  // Configure before registering this node with the control loop.
  void EnableNavgridDebug(std::string_view output_path);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  struct Cell {
    int x;
    int y;
  };
  auto IsFree(Cell cell) const -> bool;
  auto IsVisible(double x0, double y0, double x1, double y1) const -> bool;
  auto FindPath(Cell start, Cell goal) const -> std::vector<Cell>;
  auto Calculate(const PoseMessage* pose, const PoseMessage* goal,
                 std::vector<Cell>& route) const -> VelocityGoalMessage;

  std::string pose_path_;
  std::string goal_path_;
  std::string output_path_;
  std::string debug_path_;
  streamer::NavgridMessage grid_;
  MotionSettings settings_;
  std::mutex mutex_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace pathing
