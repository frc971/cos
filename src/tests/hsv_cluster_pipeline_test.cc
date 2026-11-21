#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/camera_config.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "gamepiece/hsv_cluster_tracker_node.h"
#include "localization/unambiguous_solver_node.h"
#include "utils/cv_geometry.h"

ABSL_FLAG(std::vector<std::string>, localization_logs, {}, "Three timestamped grayscale JPEG folders");
ABSL_FLAG(std::vector<std::string>, localization_configs, {}, "Three corresponding camera JSON files");
ABSL_FLAG(std::string, gamepiece_log, "", "Timestamped color JPEG folder");
ABSL_FLAG(std::string, gamepiece_config, "constants/gamepiece/camera.json", "Color camera JSON");
ABSL_FLAG(std::string, poses, "", "JSON array of {timestamp, x, y, yaw}; seconds, field meters, radians");
ABSL_FLAG(std::string, ground_truth, "", "Optional JSON array of {timestamp, clusters:[{x,y,density,in_frame_fraction}]}; field meters");
ABSL_FLAG(std::string, output_dir, "/root/debug-output/hsv-cluster-pipeline", "New or empty output folder");
ABSL_FLAG(int, timeout_seconds, 300, "Maximum replay duration including drain");
ABSL_FLAG(bool, self_test, false, "Only run input/accuracy unit tests; no hardware needed");

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using control_loop::Context;
constexpr double kScale = 80, kBallRadius = .075;

struct PoseSample { double timestamp, x, y, yaw; };
class Poses {
 public:
  explicit Poses(const Json& json) {
    if (!json.is_array() || json.empty()) throw std::invalid_argument("Poses must be a nonempty array");
    for (const auto& item : json) {
      PoseSample p{item.at("timestamp"), item.at("x"), item.at("y"), item.at("yaw")};
      if (!std::isfinite(p.timestamp) || !std::isfinite(p.x) || !std::isfinite(p.y) ||
          !std::isfinite(p.yaw) || (!samples_.empty() && p.timestamp <= samples_.back().timestamp))
        throw std::invalid_argument("Pose samples must be finite and strictly increasing");
      samples_.push_back(p);
    }
  }
  auto At(double timestamp) const -> frc::Pose2d {
    if (!std::isfinite(timestamp) || timestamp < samples_.front().timestamp || timestamp > samples_.back().timestamp)
      throw std::out_of_range("Pose samples do not cover color recording");
    auto right = std::lower_bound(samples_.begin(), samples_.end(), timestamp,
        [](const auto& p, double t) { return p.timestamp < t; });
    PoseSample p = *right;
    if (right != samples_.begin() && p.timestamp != timestamp) {
      const auto& left = *(right - 1);
      const double f = (timestamp - left.timestamp) / (p.timestamp - left.timestamp);
      p = {timestamp, std::lerp(left.x, p.x, f), std::lerp(left.y, p.y, f),
           left.yaw + f * std::remainder(p.yaw - left.yaw, 2 * std::numbers::pi)};
    }
    return {units::meter_t{p.x}, units::meter_t{p.y}, frc::Rotation2d{units::radian_t{p.yaw}}};
  }
 private:
  std::vector<PoseSample> samples_;
};

auto ReadJson(const fs::path& path) -> Json {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("Cannot open JSON: " + path.string());
  return Json::parse(input);
}

// A real log and a simulator-produced log enter through exactly the same path.
auto Timestamps(const fs::path& folder) -> std::vector<double> {
  std::vector<double> times;
  for (const auto& file : fs::directory_iterator(folder)) {
    std::string ext = file.path().extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (!file.is_regular_file() || (ext != ".jpg" && ext != ".jpeg")) continue;
    const auto stem = file.path().stem().string();
    try {
      std::size_t parsed;
      const double t = std::stod(stem, &parsed);
      if (parsed == stem.size() && std::isfinite(t)) times.push_back(t);
    } catch (const std::invalid_argument&) {} catch (const std::out_of_range&) {}
  }
  std::ranges::sort(times);
  if (times.empty() || std::adjacent_find(times.begin(), times.end()) != times.end())
    throw std::invalid_argument("Need unique timestamped JPEGs in " + folder.string());
  return times;
}

class PoseReplayNode final : public control_loop::INode {
 public:
  PoseReplayNode(const Poses& poses, double offset) : poses_(poses), offset_(offset) {}
  auto CreateCallback() -> std::function<void(const Context&)> override {
    return [this](const Context& context) {
      std::shared_ptr<localization::PositionEstimateMessage> pose;
      if (const auto* frame = context->GetMessage<camera::JpegBuffer>("color/jpeg")) {
        pose = std::make_shared<localization::PositionEstimateMessage>();
        pose->pose = frc::Pose3d{poses_.At(frame->timestamp + offset_)};
        pose->timestamp = frame->timestamp;
      }
      context->SetMessage("reference/pose", pose);
      for (const auto& callback : callbacks_) callback(context);
    };
  }
  auto GetDependencies() const -> const std::vector<control_loop::MessageDescriptor>& override { return dependencies_; }
  auto GetPublications() const -> const std::vector<control_loop::MessageDescriptor>& override { return publications_; }
  void RegisterCallback(const std::function<void(const Context&)>& callback) override { callbacks_.push_back(callback); }
 private:
  const Poses& poses_;
  double offset_;
  std::vector<control_loop::MessageDescriptor> dependencies_{{"color/jpeg", typeid(camera::JpegBuffer)}};
  std::vector<control_loop::MessageDescriptor> publications_{
      control_loop::MessageDescriptor::Publication<localization::PositionEstimateMessage>("reference/pose")};
  std::vector<std::function<void(const Context&)>> callbacks_;
};

struct Stats {
  std::size_t input = 0, dispatched = 0, processed = 0, measured = 0;
  double last = -INFINITY, seconds = 0;
  void Complete(double timestamp, double elapsed, double begin, double end) {
    CHECK_GT(timestamp, last);
    last = timestamp; ++processed;
    if (timestamp >= begin && timestamp <= end) { ++measured; seconds += elapsed; }
  }
  auto Capacity() const -> double { return seconds > 0 ? measured / seconds : 0; }
  auto Report(double wall_seconds) const -> Json {
    return {{"input_frames", input}, {"dispatched_frames", dispatched}, {"processed_frames", processed},
            {"skipped_frames", input - processed}, {"measured_frames", measured},
            {"processing_seconds", seconds}, {"processing_fps", Capacity()}, {"wall_fps", processed / wall_seconds}};
  }
};

// Rectangular Hungarian assignment; dummy columns allow unmatched truth clusters.
auto Assign(const std::vector<std::vector<double>>& costs) -> std::vector<int> {
  const int n = costs.size(), m = n ? costs.front().size() : 0;
  std::vector<double> u(n + 1), v(m + 1);
  std::vector<int> owner(m + 1), previous(m + 1), assignment(n, -1);
  for (int i = 1; i <= n; ++i) {
    owner[0] = i; int column = 0;
    std::vector<double> best(m + 1, INFINITY);
    std::vector<bool> used(m + 1);
    do {
      used[column] = true;
      const int row = owner[column]; int next = 0; double delta = INFINITY;
      for (int j = 1; j <= m; ++j) if (!used[j]) {
        const double cost = costs[row - 1][j - 1] - u[row] - v[j];
        if (cost < best[j]) { best[j] = cost; previous[j] = column; }
        if (best[j] < delta) { delta = best[j]; next = j; }
      }
      for (int j = 0; j <= m; ++j)
        if (used[j]) { u[owner[j]] += delta; v[j] -= delta; } else best[j] -= delta;
      column = next;
    } while (owner[column]);
    do { const int next = previous[column]; owner[column] = owner[next]; column = next; } while (column);
  }
  for (int j = 1; j <= m; ++j) if (owner[j]) assignment[owner[j] - 1] = j - 1;
  return assignment;
}

class Accuracy {
 public:
  explicit Accuracy(const Json& truth) {
    if (!truth.is_array() || truth.empty()) throw std::invalid_argument("Ground truth must be a nonempty array");
    for (const auto& frame : truth) {
      const double timestamp = frame.at("timestamp");
      const auto& clusters = frame.at("clusters");
      if (!std::isfinite(timestamp) || !clusters.is_array() || !truth_.emplace(timestamp, clusters).second)
        throw std::invalid_argument("Ground truth needs unique finite timestamps and cluster arrays");
      for (const auto& c : clusters) {
        const double x = c.at("x"), y = c.at("y"), density = c.at("density"), fraction = c.value("in_frame_fraction", 1.0);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(density) || density < 0 ||
            !std::isfinite(fraction) || fraction < 0 || fraction > 1)
          throw std::invalid_argument("Invalid ground truth cluster");
      }
    }
  }
  auto At(double timestamp) const -> const Json& {
    auto next = truth_.lower_bound(timestamp - 1e-6);
    if (next == truth_.end() || std::abs(next->first - timestamp) > 1e-6)
      throw std::invalid_argument("Ground truth does not cover color timestamp " + std::to_string(timestamp));
    return next->second;
  }
  void Add(double timestamp, const gamepiece::hsv_clusters_t& clusters, const frc::Pose2d& pose) {
    std::vector<Json> targets;
    for (const auto& c : At(timestamp)) {
      if (c.value("in_frame_fraction", 1.0) <= 1.0 / 3) { ++excluded_; continue; }
      targets.push_back(c);
    }
    ++frames_; expected_ += targets.size();
    // A dummy costs more than every valid match combined: maximize recall first.
    const double dummy = targets.size() + 1;
    std::vector<std::vector<double>> costs(targets.size(), std::vector<double>(clusters.size() + targets.size(), dummy));
    for (std::size_t i = 0; i < targets.size(); ++i) for (std::size_t j = 0; j < clusters.size(); ++j) {
      const auto center = pose.Translation() + clusters[j].robot_relative_translation->RotateBy(pose.Rotation());
      const double error = std::hypot(center.X().value() - targets[i].at("x").get<double>(),
                                      center.Y().value() - targets[i].at("y").get<double>());
      costs[i][j] = error <= .9 ? error : 2 * dummy;
    }
    const auto assignment = Assign(costs);
    std::vector<std::pair<double, double>> densities;
    for (std::size_t i = 0; i < targets.size(); ++i) {
      const auto j = static_cast<std::size_t>(assignment[i]);
      if (j >= clusters.size()) { ++missed_; continue; }
      errors_.push_back(costs[i][j]);
      densities.emplace_back(targets[i].at("density"), clusters[j].density);
    }
    for (std::size_t i = 0; i < densities.size(); ++i) for (std::size_t j = 0; j < i; ++j) {
      const auto [a, predicted_a] = densities[i]; const auto [b, predicted_b] = densities[j];
      if (std::abs(a - b) <= 1e-9 * std::max({1.0, a, b})) continue;
      ++pairs_; incorrect_ += (a - b) * (predicted_a - predicted_b) <= 0;
    }
  }
  auto Report() const -> Json {
    Json median = nullptr, deviation = nullptr;
    if (!errors_.empty()) {
      auto sorted = errors_; std::ranges::sort(sorted);
      const auto n = sorted.size(); median = (sorted[(n - 1) / 2] + sorted[n / 2]) / 2;
      double mean = 0, variance = 0;
      for (double e : sorted) mean += e / n;
      for (double e : sorted) variance += (e - mean) * (e - mean) / n;
      deviation = std::sqrt(variance);
    }
    return {{"frames", frames_}, {"expected_cluster_observations", expected_}, {"missed_clusters", missed_},
        {"offscreen_excluded", excluded_}, {"matched_clusters", errors_.size()}, {"centroid_error_median_m", median},
        {"centroid_error_stddev_m", deviation}, {"incorrect_density_pairs", incorrect_}, {"density_pairs", pairs_},
        {"density_order_error_fraction", pairs_ ? Json(double(incorrect_) / pairs_) : Json(nullptr)}};
  }
 private:
  std::map<double, Json> truth_;
  std::vector<double> errors_;
  std::size_t frames_ = 0, expected_ = 0, missed_ = 0, excluded_ = 0, pairs_ = 0, incorrect_ = 0;
};

auto FieldPoint(double x, double y) -> cv::Point2f {
  return {static_cast<float>(x * kScale), static_cast<float>((localization::kApriltagLayout.GetFieldWidth().value() - y) * kScale)};
}
auto FieldCanvas() -> cv::Mat {
  const auto& layout = localization::kApriltagLayout;
  cv::Mat3b canvas(static_cast<int>(layout.GetFieldWidth().value() * kScale),
                    static_cast<int>(layout.GetFieldLength().value() * kScale), cv::Vec3b{40, 40, 40});
  for (int x = 0; x <= layout.GetFieldLength().value(); ++x)
    cv::line(canvas, FieldPoint(x, 0), FieldPoint(x, layout.GetFieldWidth().value()), {70, 70, 70});
  for (int y = 0; y <= layout.GetFieldWidth().value(); ++y)
    cv::line(canvas, FieldPoint(0, y), FieldPoint(layout.GetFieldLength().value(), y), {70, 70, 70});
  for (const auto& tag : layout.GetTags())
    cv::circle(canvas, FieldPoint(tag.pose.X().value(), tag.pose.Y().value()), 3, {255, 255, 255}, cv::FILLED);
  return canvas;
}

struct Observation { cv::RotatedRect ellipse; float density; };
class Overlays {
 public:
  Overlays() : base_(FieldCanvas()), counts_(base_.size(), 0.0f) {}
  void Add(const gamepiece::hsv_clusters_t& clusters, const frc::Pose2d& pose) {
    path_.push_back(FieldPoint(pose.X().value(), pose.Y().value()));
    const auto field = [&](double x, double y) {
      return pose.Translation() + frc::Translation2d{units::meter_t{x}, units::meter_t{y}}.RotateBy(pose.Rotation());
    };
    std::set<std::size_t> ids;
    for (const auto& cluster : clusters) {
      CHECK(ids.insert(cluster.id).second);
      CHECK(cluster.robot_relative_translation.has_value());
      CHECK_EQ(cluster.ground_points.size(), cluster.ground_weights.size());
      CHECK(std::isfinite(cluster.density) && cluster.density >= 0);
      CHECK(std::isfinite(cluster.estimated_ball_count) && cluster.estimated_ball_count >= 0);
      CHECK(cv::checkRange(cluster.ground_covar));
      const auto center = field(cluster.robot_relative_translation->X().value(), cluster.robot_relative_translation->Y().value());
      CHECK(std::isfinite(center.X().value()) && std::isfinite(center.Y().value()));
      cv::Mat values, vectors;
      CHECK(cv::eigen(cv::Mat(cluster.ground_covar), values, vectors));
      CHECK_GE(values.at<float>(1), -1e-6f);
      const double angle = std::atan2(vectors.at<float>(0, 1), vectors.at<float>(0, 0)) + pose.Rotation().Radians().value();
      ellipses_[cluster.id].push_back({{FieldPoint(center.X().value(), center.Y().value()),
          {static_cast<float>(4 * kScale * std::sqrt(std::max(0.0f, values.at<float>(0)))),
           static_cast<float>(4 * kScale * std::sqrt(std::max(0.0f, values.at<float>(1))))},
          static_cast<float>(-angle * 180 / std::numbers::pi)}, cluster.density});
      for (std::size_t i = 0; i < cluster.ground_points.size(); ++i) {
        const auto& p = cluster.ground_points[i];
        CHECK(std::isfinite(p.x) && std::isfinite(p.y));
        CHECK(std::isfinite(cluster.ground_weights[i]) && cluster.ground_weights[i] >= 0);
        const auto point = field(p.x, p.y);
        const auto pixel = FieldPoint(point.X().value(), point.Y().value());
        if (pixel.x >= 0 && pixel.x < counts_.cols && pixel.y >= 0 && pixel.y < counts_.rows)
          counts_(static_cast<int>(pixel.y), static_cast<int>(pixel.x)) += cluster.ground_weights[i];
      }
      ++observations;
    }
  }
  void Write(const fs::path& directory) {
    const int radius = std::lround(kBallRadius * kScale);
    cv::Mat1f disk(2 * radius + 1, 2 * radius + 1, 0.0f);
    cv::circle(disk, {radius, radius}, radius, 1, cv::FILLED); disk /= cv::sum(disk)[0];
    cv::filter2D(counts_, counts_, -1, disk, {-1, -1}, 0, cv::BORDER_CONSTANT);
    double peak; cv::minMaxLoc(counts_, nullptr, &peak);
    cv::Mat intensity, levels, colors, blended, heatmap = base_.clone(), overlay = base_.clone();
    cv::log(counts_ + 1, intensity);
    intensity.convertTo(levels, CV_8U, peak > 0 ? 255 / std::log1p(peak) : 0);
    cv::applyColorMap(levels, colors, cv::COLORMAP_TURBO);
    cv::addWeighted(colors, .85, base_, .15, 0, blended);
    blended.copyTo(heatmap, counts_ > .001f);
    for (std::size_t i = 1; i < path_.size(); ++i) {
      cv::line(overlay, path_[i - 1], path_[i], {230, 230, 230}, 1, cv::LINE_AA);
      cv::line(heatmap, path_[i - 1], path_[i], {230, 230, 230}, 1, cv::LINE_AA);
    }
    for (auto& [id, samples] : ellipses_) {
      auto median = samples.begin() + samples.size() / 2;
      std::nth_element(samples.begin(), median, samples.end(), [](const auto& a, const auto& b) { return a.ellipse.size.area() < b.ellipse.size.area(); });
      cv::Mat3b hsv(1, 1, cv::Vec3b{static_cast<uchar>((id * 67) % 180), 220, 255}), bgr;
      cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
      const auto c = bgr(0, 0); const cv::Scalar color{double(c[0]), double(c[1]), double(c[2])};
      auto filled = overlay.clone();
      cv::ellipse(filled, median->ellipse, color, cv::FILLED, cv::LINE_AA);
      cv::addWeighted(filled, .3, overlay, .7, 0, overlay);
      cv::ellipse(overlay, median->ellipse, color, 1, cv::LINE_AA);
      cv::putText(overlay, cv::format("%zu: %.1f balls/m2", id, median->density), median->ellipse.center,
                  cv::FONT_HERSHEY_SIMPLEX, .45, {255, 255, 255}, 1, cv::LINE_AA);
    }
    cv::putText(heatmap, cv::format("%zu frames | accumulated support | peak %.3f (log scale)", path_.size(), peak),
                {20, 28}, cv::FONT_HERSHEY_SIMPLEX, .6, {255, 255, 255}, 1, cv::LINE_AA);
    CHECK(cv::imwrite((directory / "overall_heatmap.png").string(), heatmap));
    CHECK(cv::imwrite((directory / "field_overlay.png").string(), overlay));
  }
  std::size_t observations = 0;
 private:
  cv::Mat base_;
  cv::Mat1f counts_;
  std::vector<cv::Point2f> path_;
  std::map<std::size_t, std::vector<Observation>> ellipses_;
};

auto RunReplay() -> int {
  auto folders = absl::GetFlag(FLAGS_localization_logs);
  auto configs = absl::GetFlag(FLAGS_localization_configs);
  if (folders.size() != 3 || configs.size() != 3 || absl::GetFlag(FLAGS_gamepiece_log).empty())
    throw std::invalid_argument("Provide --localization_logs=DIR1,DIR2,DIR3 --localization_configs=JSON1,JSON2,JSON3 --gamepiece_log=DIR --poses=JSON");
  folders.push_back(absl::GetFlag(FLAGS_gamepiece_log)); configs.push_back(absl::GetFlag(FLAGS_gamepiece_config));
  const Poses poses(ReadJson(absl::GetFlag(FLAGS_poses)));
  std::unique_ptr<Accuracy> accuracy;
  if (!absl::GetFlag(FLAGS_ground_truth).empty())
    accuracy = std::make_unique<Accuracy>(ReadJson(absl::GetFlag(FLAGS_ground_truth)));
  std::array<std::vector<double>, 4> times;
  std::array<Stats, 4> stats;
  std::array<cv::Size, 4> sizes;
  double offset = INFINITY, begin = -INFINITY, end = INFINITY;
  for (std::size_t i = 0; i < 4; ++i) {
    times[i] = Timestamps(folders[i]); stats[i].input = times[i].size();
    offset = std::min(offset, times[i].front());
    begin = std::max(begin, times[i].front()); end = std::min(end, times[i].back());
    CHECK(fs::is_regular_file(configs[i])) << configs[i];
    const auto config = ReadJson(configs[i]);
    sizes[i] = {config.at("width").get<int>(), config.at("height").get<int>()};
    CHECK_GT(sizes[i].width, 0); CHECK_GT(sizes[i].height, 0);
  }
  if (begin >= end) throw std::invalid_argument("All four camera recordings must overlap");
  for (const auto& stream : times)
    if (std::ranges::count_if(stream, [&](double t) { return t >= begin && t <= end; }) < 2)
      throw std::invalid_argument("Need at least two frames per camera in the shared interval");
  poses.At(times[3].front()); poses.At(times[3].back());
  if (accuracy) for (double timestamp : times[3]) accuracy->At(timestamp);
  const fs::path output = absl::GetFlag(FLAGS_output_dir);
  if (fs::exists(output) && !fs::is_empty(output)) throw std::invalid_argument("Output directory must be new or empty");
  fs::create_directories(output);
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);
  begin -= offset; end -= offset;

  std::mutex mutex;
  std::condition_variable completion;
  Overlays overlays;
  double tracker_seconds = 0;
  std::size_t tracker_frames = 0, localization_poses = 0;
  std::uint64_t last_context = 0;
  control_loop::ThreadPool pool(6);
  control_loop::ControlLoop localization_loop(std::chrono::milliseconds(1)), gamepiece_loop(std::chrono::milliseconds(1));
  auto solver = std::make_shared<localization::UnambiguousSolverNode>("stress/pose");
  auto pose_node = std::make_shared<PoseReplayNode>(poses, offset);
  auto tracker = std::make_shared<gamepiece::HSVClusterTrackerNode>("color/decoded", "clusters",
      camera::Intrinsics{configs[3]}.ToMatrix(),
      utils::EigenToCvMat(camera::Extrinsics{configs[3]}.ToRobotToCamera<frc::Transform3d>().ToMatrix()),
      static_cast<std::size_t>(sizes[3].width) * sizes[3].height, "reference/pose");
  std::array<std::shared_ptr<camera::UVCDiskCameraNode>, 4> cameras;
  // Initialize GPU nodes before starting the replay clock/camera threads.
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string prefix = i == 3 ? "color" : "gray" + std::to_string(i);
    auto decoder = std::make_shared<camera::NvjpegDecodeNode>(prefix + "/jpeg", prefix + "/decoded",
        i == 3 ? NVJPEG_OUTPUT_BGRI : NVJPEG_OUTPUT_Y, pool);
    if (i == 3) {
      // Publish reference pose synchronously before dispatching color decode.
      gamepiece_loop.RegisterNode(pose_node);
      decoder->RegisterCallback([](const Context& context) {
        context->SetMessage("tracker/start", std::make_shared<control_loop::ValueMessage<Clock::time_point>>(Clock::now()));
      });
      gamepiece_loop.RegisterNode(decoder);
      gamepiece_loop.RegisterNode(tracker);
    } else {
      auto detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(prefix + "/decoded", prefix + "/tags", configs[i], pool);
      detector->WarmUp();
      localization_loop.RegisterNode(decoder); localization_loop.RegisterNode(detector);
      solver->AddCamera(prefix + "/tags", camera::Intrinsics{configs[i]}, camera::Extrinsics{configs[i]}, localization_loop);
      solver->AddCameraTimestamp(prefix + "/jpeg");
    }
  }
  solver->RegisterCallback([&](const Context& context) {
    const auto now = Clock::now();
    std::scoped_lock lock(mutex);
    if (last_context == context->id) return;
    last_context = context->id;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto prefix = "gray" + std::to_string(i);
      if (const auto* frame = context->GetMessage<camera::JpegBuffer>(prefix + "/jpeg")) {
        const auto* decoded = context->GetMessage<camera::DecodedJpegBuffer>(prefix + "/decoded");
        CHECK(decoded);
        CHECK_EQ(decoded->width, sizes[i].width); CHECK_EQ(decoded->height, sizes[i].height);
        CHECK(context->GetMessage<apriltag::TagDetections>(prefix + "/tags"));
        stats[i].Complete(frame->timestamp, std::chrono::duration<double>(now - context->start).count(), begin, end);
      }
    }
    if (context->GetMessage<localization::PositionEstimateMessage>("stress/pose")) ++localization_poses;
    completion.notify_one();
  });
  localization_loop.RegisterNode(solver);
  tracker->RegisterCallback([&](const Context& context) {
    const auto now = Clock::now();
    const auto* clusters = context->GetMessage<control_loop::ValueMessage<gamepiece::hsv_clusters_t>>("clusters");
    if (!clusters) return;
    const auto* frame = context->GetMessage<camera::DecodedJpegBuffer>("color/decoded");
    const auto* pose = context->GetMessage<localization::PositionEstimateMessage>("reference/pose");
    CHECK(frame && pose);
    CHECK_EQ(frame->width, sizes[3].width); CHECK_EQ(frame->height, sizes[3].height);
    const auto started = context->GetMessage<control_loop::ValueMessage<Clock::time_point>>("tracker/start")->value;
    if (frame->timestamp >= begin && frame->timestamp <= end) {
      tracker_seconds += std::chrono::duration<double>(now - started).count(); ++tracker_frames;
    }
    overlays.Add(clusters->value, pose->pose.ToPose2d());
    if (accuracy) accuracy->Add(frame->timestamp + offset, clusters->value, pose->pose.ToPose2d());
    std::scoped_lock lock(mutex);
    stats[3].Complete(frame->timestamp, std::chrono::duration<double>(now - context->start).count(), begin, end);
    completion.notify_one();
  });

  control_loop::RioClock::EnableSimulation();
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string jpeg = i == 3 ? "color/jpeg" : "gray" + std::to_string(i) + "/jpeg";
    cameras[i] = std::make_shared<camera::UVCDiskCameraNode>(folders[i], jpeg, offset);
    cameras[i]->RegisterCallback([&, i, jpeg](const Context& context) {
      if (context->GetMessage<camera::JpegBuffer>(jpeg)) { std::scoped_lock lock(mutex); ++stats[i].dispatched; }
    });
    (i == 3 ? gamepiece_loop : localization_loop).RegisterDependencyNode(cameras[i]);
  }
  const auto started = Clock::now();
  localization_loop.Start(); gamepiece_loop.Start();
  bool drained;
  {
    std::unique_lock lock(mutex);
    drained = completion.wait_for(lock, std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)), [&] {
      for (std::size_t i = 0; i < 4; ++i) if (stats[i].last != times[i].back() - offset) return false;
      return true;
    });
  }
  // A camera's EOF requests global stop; wait for every terminal frame instead.
  gamepiece_loop.Stop(); localization_loop.Stop(); pool.Shutdown();
  const double wall_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  bool passed = drained && overlays.observations > 0;
  Json summary{{"completed", drained}, {"cluster_observations", overlays.observations}, {"localization_poses", localization_poses},
      {"tracker_fps", tracker_seconds > 0 ? tracker_frames / tracker_seconds : 0}, {"wall_seconds", wall_seconds},
      {"measurement_interval", {begin + offset, end + offset}}, {"cameras", Json::array()}};
  for (std::size_t i = 0; i < 4; ++i) {
    const bool ok = stats[i].processed == stats[i].dispatched && stats[i].measured >= 2 && stats[i].Capacity() > 30;
    passed &= ok;
    auto report = stats[i].Report(wall_seconds); report["folder"] = folders[i]; report["passed"] = ok;
    summary["cameras"].push_back(report);
    LOG(INFO) << folders[i] << ": " << report.dump();
  }
  summary["passed"] = passed;
  if (accuracy) { summary["accuracy"] = accuracy->Report(); LOG(INFO) << "Accuracy: " << summary["accuracy"].dump(); }
  overlays.Write(output);
  std::ofstream result(output / "summary.json"); result << summary.dump(2) << '\n'; result.flush(); CHECK(result.good());
  LOG(INFO) << "Replay " << (passed ? "passed" : "failed") << "; tracker FPS=" << summary["tracker_fps"];
  return passed ? 0 : 1;
}

TEST(PoseReplayTest, InterpolatesTranslationAndShortestYawIncludingEndpoints) {
  const Poses poses(Json::parse(R"([{"timestamp":2,"x":1,"y":3,"yaw":3.1},{"timestamp":4,"x":5,"y":7,"yaw":-3.1}])"));
  EXPECT_DOUBLE_EQ(poses.At(2).X().value(), 1); EXPECT_DOUBLE_EQ(poses.At(4).Y().value(), 7);
  EXPECT_DOUBLE_EQ(poses.At(3).X().value(), 3); EXPECT_DOUBLE_EQ(poses.At(3).Y().value(), 5);
  EXPECT_NEAR(std::abs(poses.At(3).Rotation().Radians().value()), std::numbers::pi, 1e-12);
  EXPECT_THROW(poses.At(1), std::out_of_range);
  EXPECT_THROW(poses.At(5), std::out_of_range);
}
TEST(PoseReplayTest, RejectsMalformedNonfiniteAndUnorderedSamples) {
  EXPECT_THROW(Poses(Json::array()), std::invalid_argument);
  EXPECT_THROW(Poses(Json::parse(R"([{"timestamp":0}])")), Json::exception);
  const Json sample{{"timestamp", 1}, {"x", 0}, {"y", 0}, {"yaw", 0}};
  EXPECT_THROW(Poses(Json::array({sample, sample})), std::invalid_argument);
  auto earlier = sample; earlier["timestamp"] = 0;
  EXPECT_THROW(Poses(Json::array({sample, earlier})), std::invalid_argument);
  auto invalid = sample; invalid["yaw"] = INFINITY;
  EXPECT_THROW(Poses(Json::array({invalid})), std::invalid_argument);
  const Poses single(Json::array({sample}));
  EXPECT_DOUBLE_EQ(single.At(1).X().value(), 0);
  EXPECT_THROW(single.At(NAN), std::out_of_range);
}
TEST(AccuracyTest, MatchesOneToOneBeforeMinimizingDistance) {
  EXPECT_EQ(Assign({{.1, .8, 3, 3}, {.2, 6, 3, 3}}), (std::vector<int>{1, 0}));
  EXPECT_TRUE(Assign({}).empty());
}
TEST(AccuracyTest, ReportsFieldErrorsMissesVisibilityAndDensityInversions) {
  Accuracy accuracy(Json::parse(R"([{"timestamp":10,"clusters":[
    {"x":2,"y":9,"density":10},{"x":2,"y":12,"density":20},
    {"x":100,"y":100,"density":30},{"x":50,"y":50,"density":40,"in_frame_fraction":0.3333333333333333}]}])"));
  gamepiece::hsv_clusters_t clusters(2);
  clusters[0].robot_relative_translation = frc::Translation2d{6_m, 0_m}; clusters[0].density = 3;
  clusters[1].robot_relative_translation = frc::Translation2d{9.6_m, 0_m}; clusters[1].density = 2;
  accuracy.Add(10, clusters, frc::Pose2d{2_m, 3_m, frc::Rotation2d{units::radian_t{std::numbers::pi / 2}}});
  const auto r = accuracy.Report();
  EXPECT_EQ(r.at("expected_cluster_observations"), 3); EXPECT_EQ(r.at("missed_clusters"), 1);
  EXPECT_EQ(r.at("offscreen_excluded"), 1); EXPECT_EQ(r.at("matched_clusters"), 2);
  EXPECT_NEAR(r.at("centroid_error_median_m").get<double>(), .3, 1e-9);
  EXPECT_NEAR(r.at("centroid_error_stddev_m").get<double>(), .3, 1e-9);
  EXPECT_EQ(r.at("incorrect_density_pairs"), 1); EXPECT_EQ(r.at("density_pairs"), 1);
  EXPECT_EQ(r.at("density_order_error_fraction"), 1);
}
TEST(AccuracyTest, DistinguishesUnmeasuredErrorsAndDensityTies) {
  Accuracy accuracy(Json::parse(R"([{"timestamp":0,"clusters":[
    {"x":0,"y":0,"density":10},{"x":3,"y":0,"density":20}]}])"));
  accuracy.Add(0, {}, {});
  EXPECT_TRUE(accuracy.Report().at("centroid_error_median_m").is_null());
  EXPECT_TRUE(accuracy.Report().at("density_order_error_fraction").is_null());
  gamepiece::hsv_clusters_t clusters(2);
  for (int i = 0; i < 2; ++i) {
    clusters[i].robot_relative_translation = frc::Translation2d{units::meter_t{3.0 * i}, 0_m}; clusters[i].density = 1;
  }
  accuracy.Add(0, clusters, {});
  EXPECT_EQ(accuracy.Report().at("incorrect_density_pairs"), 1);
  Accuracy equal(Json::parse(R"([{"timestamp":0,"clusters":[
    {"x":0,"y":0,"density":10},{"x":3,"y":0,"density":10}]}])"));
  equal.Add(0, clusters, {}); EXPECT_EQ(equal.Report().at("density_pairs"), 0);
}
TEST(AccuracyTest, RejectsInvalidOrMissingTruth) {
  const Json frame{{"timestamp", 1}, {"clusters", Json::array()}};
  EXPECT_THROW(Accuracy(Json::array()), std::invalid_argument);
  EXPECT_THROW(Accuracy(Json::array({frame, frame})), std::invalid_argument);
  Accuracy accuracy(Json::array({frame})); EXPECT_THROW(accuracy.At(2), std::invalid_argument);
  auto invalid = frame; invalid["clusters"] = Json::array({{{"x", 0}, {"y", 0}, {"density", -1}}});
  EXPECT_THROW(Accuracy(Json::array({invalid})), std::invalid_argument);
}
}  // namespace

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv); absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  if (absl::GetFlag(FLAGS_self_test)) return RUN_ALL_TESTS();
  int result;
  try { result = RunReplay(); }
  catch (const std::exception& error) { LOG(ERROR) << error.what(); result = 1; }
  // Match the existing replay tests after node destruction: VPI/CUDA exit
  // finalizers conflict on Jetson; workers and outputs have already drained.
  std::fflush(nullptr); std::_Exit(result);
}
