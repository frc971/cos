#include "gamepiece/ball_pathing_node.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "camera/nvjpeg_decode_node.h"
#include "gamepiece/unet_preprocess.h"
#include "utils/cv_geometry.h"

namespace gamepiece {
namespace {

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 400;
constexpr size_t kInputElements = kInputWidth * kInputHeight;

auto LoadEngine(const std::string& path) -> std::vector<char> {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("U-Net TensorRT engine not found: " + path);
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

auto TensorVolume(const nvinfer1::Dims& dimensions) -> size_t {
  size_t result = 1;
  for (int i = 0; i < dimensions.nbDims; ++i) {
    CHECK_GT(dimensions.d[i], 0) << "Dynamic TensorRT tensors are unsupported";
    result *= static_cast<size_t>(dimensions.d[i]);
  }
  return result;
}

}  // namespace

BallPathingNode::BallPathingNode(
    std::string_view input_path, std::string_view output_path,
    const std::string& model_path, const std::string& camera_config_path,
    control_loop::ThreadPool& thread_pool, BallPathingConfig config)
    : input_path_(input_path),
      output_path_(output_path),
      thread_pool_(thread_pool),
      config_(config),
      intrinsics_(camera_config_path),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          BallVelocityGoalMessage>(output_path_)}) {
  CHECK_GE(config_.probability_threshold, 0.0F);
  CHECK_LE(config_.probability_threshold, 1.0F);
  CHECK_GT(config_.mask_sample_stride, 0);
  CHECK_GT(config_.maximum_range_m, 0.0);
  CHECK_GT(config_.cluster_radius_m, 0.0);
  CHECK_GT(config_.grid_resolution_m, 0.0);
  CHECK_GT(config_.nominal_ball_area_m2, 0.0);
  CHECK_GT(config_.forward_speed_mps, 0.0);
  CHECK_GT(config_.yaw_gain_per_s, 0.0);
  CHECK_GT(config_.maximum_yaw_velocity_rad_per_s, 0.0);
  CHECK_GE(config_.switch_evidence_ratio, 1.0);
  CHECK_GE(config_.hold_duration.count(), 0.0);

  const camera::Extrinsics extrinsics(camera_config_path);
  camera_to_robot_ = utils::EigenToCvMat(
      extrinsics.ToRobotToCamera<frc::Transform3d>().ToMatrix());

  const std::vector<char> engine_data = LoadEngine(model_path);
  runtime_ = nvinfer1::createInferRuntime(*this);
  CHECK(runtime_ != nullptr);
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  CHECK(engine_ != nullptr) << "Unable to deserialize " << model_path;
  CHECK_EQ(engine_->getNbIOTensors(), 2);
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_tensor_name_ = name;
    } else {
      output_tensor_name_ = name;
    }
  }
  CHECK(!input_tensor_name_.empty());
  CHECK(!output_tensor_name_.empty());
  CHECK_EQ(TensorVolume(engine_->getTensorShape(input_tensor_name_.c_str())),
           kInputElements);
  CHECK_EQ(TensorVolume(engine_->getTensorShape(output_tensor_name_.c_str())),
           kInputElements);
  CHECK_EQ(engine_->getTensorDataType(input_tensor_name_.c_str()),
           nvinfer1::DataType::kFLOAT);
  CHECK_EQ(engine_->getTensorDataType(output_tensor_name_.c_str()),
           nvinfer1::DataType::kFLOAT);

  context_ = engine_->createExecutionContext();
  CHECK(context_ != nullptr);
  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&input_buffer_),
                      kInputElements * sizeof(float)),
           cudaSuccess);
  CHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&output_buffer_),
                      kInputElements * sizeof(float)),
           cudaSuccess);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
}

BallPathingNode::~BallPathingNode() {
  delete context_;
  delete engine_;
  delete runtime_;
  if (output_buffer_ != nullptr) cudaFree(output_buffer_);
  if (input_buffer_ != nullptr) cudaFree(input_buffer_);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void BallPathingNode::log(Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) {
    LOG(WARNING) << "TensorRT U-Net: " << message;
  }
}

auto BallPathingNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto frame =
        context->GetSharedMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr ||
        frame->width <= 0 || frame->height <= 0) {
      std::scoped_lock lock(inference_mutex_);
      context->SetMessage(
          output_path_,
          std::make_shared<BallVelocityGoalMessage>(NoDetection(0.0)));
      for (const auto& callback : callbacks_) callback(context);
      return;
    }
    thread_pool_.Submit(
        [this, context, frame]() -> void {
          CHECK_EQ(frame->output_format, NVJPEG_OUTPUT_Y);
          CHECK_GE(frame->stride, static_cast<size_t>(frame->width));
          const cv::cuda::GpuMat image(frame->height, frame->width, CV_8UC1,
                                       frame->destination.channel[0],
                                       frame->stride);
          BallVelocityGoalMessage result;
          {
            std::scoped_lock lock(inference_mutex_);
            result = Process(image, frame->timestamp);
          }
          context->SetMessage(
              output_path_,
              std::make_shared<BallVelocityGoalMessage>(std::move(result)));
          for (const auto& callback : callbacks_) callback(context);
        },
        context->id);
  };
}

void BallPathingNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto BallPathingNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto BallPathingNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto BallPathingNode::RunInference(const cv::cuda::GpuMat& image,
                                   cv::Size original_size) -> cv::Mat1b {
  CHECK_EQ(PreprocessUnetImage(image.data, image.cols, image.rows, image.step,
                               input_buffer_, kInputWidth, kInputHeight,
                               stream_),
           cudaSuccess);
  CHECK(context_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_));
  CHECK(context_->setTensorAddress(output_tensor_name_.c_str(), output_buffer_));
  CHECK(context_->enqueueV3(stream_));
  std::vector<float> probabilities(kInputElements);
  CHECK_EQ(cudaMemcpyAsync(probabilities.data(), output_buffer_,
                           probabilities.size() * sizeof(float),
                           cudaMemcpyDeviceToHost, stream_),
           cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  cv::Mat1f small(kInputHeight, kInputWidth, probabilities.data());
  cv::Mat1f restored;
  cv::resize(small, restored, original_size, 0, 0, cv::INTER_LINEAR);
  restored.rowRange(static_cast<int>(restored.rows * (2.0 / 3.0)),
                    restored.rows) = 0.0F;
  return restored > config_.probability_threshold;
}

auto BallPathingNode::ProjectMask(const cv::Mat1b& mask) const
    -> std::vector<internal::ProjectedMaskSample> {
  std::vector<cv::Point2f> pixels;
  const int stride = config_.mask_sample_stride;
  for (int y = 0; y + stride < mask.rows; y += stride) {
    for (int x = 0; x + stride < mask.cols; x += stride) {
      if (mask(y, x) == 0) continue;
      pixels.emplace_back(x, y);
      pixels.emplace_back(x + stride, y);
      pixels.emplace_back(x, y + stride);
    }
  }
  if (pixels.empty()) return {};
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(pixels, normalized, intrinsics_.ToMatrix(),
                      intrinsics_.ToDistortionCoefficients());
  const cv::Vec3d origin{camera_to_robot_(0, 3), camera_to_robot_(1, 3),
                         camera_to_robot_(2, 3)};
  const cv::Matx33d rotation(camera_to_robot_(0, 0), camera_to_robot_(0, 1),
                             camera_to_robot_(0, 2), camera_to_robot_(1, 0),
                             camera_to_robot_(1, 1), camera_to_robot_(1, 2),
                             camera_to_robot_(2, 0), camera_to_robot_(2, 1),
                             camera_to_robot_(2, 2));
  const auto project = [&](const cv::Point2f point)
      -> std::optional<cv::Point2d> {
    // OpenCV camera axes (right, down, forward) to WPILib camera axes
    // (forward, left, up), followed by the calibrated camera pose.
    const cv::Vec3d direction = rotation * cv::Vec3d{1.0, -point.x, -point.y};
    if (direction[2] >= -1e-6) return std::nullopt;
    const double scale = -origin[2] / direction[2];
    if (scale <= 0.0) return std::nullopt;
    const cv::Vec3d intersection = origin + scale * direction;
    const cv::Point2d robot_point(intersection[0], intersection[1]);
    const double range = cv::norm(robot_point);
    // The color camera may face forward or backward. Its calibrated pose and
    // optical ray already determine the visible half-plane, so only constrain
    // total range here rather than rejecting valid rear-facing targets.
    if (range <= config_.maximum_range_m) {
      return robot_point;
    }
    return std::nullopt;
  };

  std::vector<internal::ProjectedMaskSample> result;
  result.reserve(normalized.size() / 3);
  for (size_t i = 0; i + 2 < normalized.size(); i += 3) {
    const auto center = project(normalized[i]);
    const auto right = project(normalized[i + 1]);
    const auto down = project(normalized[i + 2]);
    if (!center.has_value() || !right.has_value() || !down.has_value()) {
      continue;
    }
    const cv::Point2d across = *right - *center;
    const cv::Point2d along = *down - *center;
    const double area = std::abs(across.x * along.y - across.y * along.x);
    if (area > 0.0 && std::isfinite(area)) {
      result.push_back({.position = *center, .evidence_m2 = area});
    }
  }
  return result;
}

auto BallPathingNode::NoDetection(double timestamp)
    -> BallVelocityGoalMessage {
  const auto now = std::chrono::steady_clock::now();
  if (have_target_ && now - last_seen_ <= config_.hold_duration) {
    BallVelocityGoalMessage held = last_goal_;
    held.held = true;
    held.timestamp = timestamp;
    return held;
  }
  have_target_ = false;
  tracked_cluster_ = {};
  last_goal_ = {};
  last_goal_.timestamp = timestamp;
  return last_goal_;
}

auto BallPathingNode::Process(const cv::cuda::GpuMat& image, double timestamp)
    -> BallVelocityGoalMessage {
  const cv::Mat1b mask = RunInference(image, image.size());
  const std::vector<internal::ProjectedMaskSample> samples = ProjectMask(mask);
  const std::vector<internal::BallCluster> clusters =
      internal::BuildBallClusters(samples, config_.cluster_radius_m,
                                  config_.grid_resolution_m);
  const std::optional<internal::BallCluster> selected =
      internal::ChooseBallCluster(
          clusters,
          have_target_ ? std::optional<internal::BallCluster>(tracked_cluster_)
                       : std::nullopt,
          config_.cluster_radius_m, config_.switch_evidence_ratio);
  if (!selected.has_value()) return NoDetection(timestamp);

  tracked_cluster_ = *selected;
  have_target_ = true;
  last_seen_ = std::chrono::steady_clock::now();
  const internal::BallDriveVelocity velocity = internal::ComputeDriveVelocity(
      selected->centroid, config_.forward_speed_mps, config_.yaw_gain_per_s,
      config_.maximum_yaw_velocity_rad_per_s);
  last_goal_ = {};
  last_goal_.valid = true;
  last_goal_.vx_mps = velocity.vx_mps;
  last_goal_.vy_mps = velocity.vy_mps;
  last_goal_.omega_rad_per_s = velocity.omega_rad_per_s;
  last_goal_.target_x_m = selected->centroid.x;
  last_goal_.target_y_m = selected->centroid.y;
  last_goal_.ball_evidence_m2 = selected->evidence_m2;
  last_goal_.estimated_ball_count = std::max(
      1, static_cast<int>(std::lround(selected->evidence_m2 /
                                      config_.nominal_ball_area_m2)));
  last_goal_.timestamp = timestamp;
  return last_goal_;
}

}  // namespace gamepiece
