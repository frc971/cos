#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "camera/camera_config.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/ball_pathing_logic.h"
#include "logging/log_registration.h"

namespace gamepiece {

struct BallVelocityGoalMessage final : public control_loop::IMessage {
  bool valid = false;
  bool held = false;
  double vx_mps = 0.0;
  double vy_mps = 0.0;
  double target_x_m = 0.0;
  double target_y_m = 0.0;
  double omega_rad_per_s = 0.0;
  int estimated_ball_count = 0;
  double ball_evidence_m2 = 0.0;
  double timestamp = 0.0;

  auto GetType() -> const std::type_info& override {
    return typeid(BallVelocityGoalMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }

  LOG_FIELDS(BallVelocityGoalMessage, valid, held, vx_mps, vy_mps, target_x_m,
             target_y_m, omega_rad_per_s, estimated_ball_count,
             ball_evidence_m2, timestamp)
};

struct BallPathingConfig {
  float probability_threshold = 0.5F;
  int mask_sample_stride = 4;
  double maximum_range_m = 8.0;
  double cluster_radius_m = 1.0;
  double grid_resolution_m = 0.20;
  double nominal_ball_area_m2 = 0.018;
  double forward_speed_mps = 2.5;
  double yaw_gain_per_s = 3.0;
  double maximum_yaw_velocity_rad_per_s = 5.026548245743669;
  double switch_evidence_ratio = 1.25;
  std::chrono::duration<double> hold_duration{0.5};
};

// Runs the grayscale U-Net and converts its segmentation into a robot-relative
// translational velocity goal. Robot coordinates are +X forward and +Y left.
class BallPathingNode final : public control_loop::INode,
                              private nvinfer1::ILogger {
 public:
  BallPathingNode(std::string_view input_path, std::string_view output_path,
                  const std::string& model_path,
                  const std::string& camera_config_path,
                  control_loop::ThreadPool& thread_pool,
                  BallPathingConfig config = {});
  ~BallPathingNode() override;

  BallPathingNode(const BallPathingNode&) = delete;
  auto operator=(const BallPathingNode&) -> BallPathingNode& = delete;

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  void log(Severity severity, const char* message) noexcept override;
  auto Process(const cv::cuda::GpuMat& image, double timestamp)
      -> BallVelocityGoalMessage;
  auto RunInference(const cv::cuda::GpuMat& image, cv::Size original_size)
      -> cv::Mat1b;
  auto ProjectMask(const cv::Mat1b& mask) const
      -> std::vector<internal::ProjectedMaskSample>;
  auto NoDetection(double timestamp) -> BallVelocityGoalMessage;

  std::string input_path_;
  std::string output_path_;
  control_loop::ThreadPool& thread_pool_;
  BallPathingConfig config_;
  camera::Intrinsics intrinsics_;
  cv::Matx44d camera_to_robot_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex inference_mutex_;

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  float* input_buffer_ = nullptr;
  float* output_buffer_ = nullptr;
  std::string input_tensor_name_;
  std::string output_tensor_name_;

  bool have_target_ = false;
  internal::BallCluster tracked_cluster_;
  BallVelocityGoalMessage last_goal_;
  std::chrono::steady_clock::time_point last_seen_{};
};

}  // namespace gamepiece
