// Offline mask ablation: identical native analytical geometry, fresh state per image.
#include <fstream>
#include <iostream>
#include <numbers>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include "camera/camera_config.h"
#include "gamepiece/hsv_cluster_tracker_node.h"
#include "utils/cv_geometry.h"

namespace gamepiece {
struct HSVClusterTrackerNodeTestPeer {
  static auto Run(HSVClusterTrackerNode& tracker, const cv::Mat& image,
                  const cv::Mat& mask, const frc::Pose2d& pose) -> nlohmann::json {
    tracker.ProcessAnalytical(image, pose, mask);
    const auto rotation = tracker.camera_extrinsics_cv_.get_minor<3, 3>(0, 0);
    const auto project = [&](const cv::Vec3f& ray) {
      const auto& k = tracker.camera_intrinsics_;
      return nlohmann::json::array({ray[0] / ray[2] * k(0, 0) + k(0, 2),
                                    ray[1] / ray[2] * k(1, 1) + k(1, 2)});
    };
    const auto center = [&](cv::Point2f p) {
      return rotation.t() * (cv::Vec3f{-p.y, -.075f, p.x} - tracker.camera_position_);
    };
    nlohmann::json clusters = nlohmann::json::array();
    for (const auto& c : *tracker.GetClusters()) {
      nlohmann::json spheres = nlohmann::json::array(), tangents = nlohmann::json::array();
      for (const auto& p : c.verified_ball_centers) {
        const auto ray = center(p), axis = ray / cv::norm(ray);
        auto u = axis.cross(cv::Vec3f{0, 1, 0}); u /= cv::norm(u);
        const auto v = axis.cross(u); const float s = .075f / cv::norm(ray);
        nlohmann::json outline = nlohmann::json::array();
        for (int j = 0; j < 32; ++j) {
          const float a = 2 * std::numbers::pi_v<float> * j / 32;
          outline.push_back(project(std::sqrt(1-s*s)*axis + s*(std::cos(a)*u + std::sin(a)*v)));
        }
        spheres.push_back({{"center", project(ray)}, {"outline", outline}, {"ground", {p.x, p.y}}});
      }
      for (std::size_t j = 0; j < c.ground_points.size(); ++j)
        tangents.push_back({{"center", project(center(c.ground_points[j]))}, {"weight", c.ground_weights[j]}});
      const auto p = *c.robot_relative_translation;
      clusters.push_back({{"count", c.estimated_ball_count}, {"density", c.density},
        {"ground", {p.X().value(), p.Y().value()}}, {"spheres", spheres}, {"tangents", tangents},
        {"residual", {c.residual_ball_evidence[0], c.residual_ball_evidence[1]}}});
    }
    nlohmann::json camera = nlohmann::json::array();
    for (int row = 0; row < 4; ++row) {
      camera.push_back(nlohmann::json::array());
      for (int col = 0; col < 4; ++col) camera.back().push_back(tracker.camera_extrinsics_cv_(row,col));
    }
    return {{"clusters", clusters}, {"camera_extrinsics_cv", camera}};
  }
};
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("Usage: analytical_mask_eval camera.json cases.json output.json");
    const camera::Intrinsics k{argv[1]};
    const auto extrinsics = utils::EigenToCvMat(camera::Extrinsics{argv[1]}.ToRobotToCamera<frc::Transform3d>().ToMatrix());
    nlohmann::json cases; std::ifstream(argv[2]) >> cases;
    nlohmann::json results = nlohmann::json::array();
    for (const auto& item : cases) {
      std::cerr << item.at("frame") << ' ' << item.at("method") << '\n';
      const auto image = cv::imread(item.at("image").get<std::string>());
      const auto path = item.at("mask").get<std::string>();
      const auto mask = path.empty() ? cv::Mat{} : cv::imread(path, cv::IMREAD_GRAYSCALE);
      if (image.empty() || (!path.empty() && mask.empty())) throw std::runtime_error("Cannot decode evaluation input");
      gamepiece::HSVClusterTrackerNode tracker("", "", k.ToMatrix(), extrinsics, image.total(),
          "", gamepiece::DensityHistory::kNone);
      auto result = item;
      const auto p = item.at("pose");
      const frc::Pose2d pose{units::meter_t{p[0].get<double>()}, units::meter_t{p[1].get<double>()},
                             frc::Rotation2d{units::radian_t{p[2].get<double>()}}};
      result.update(gamepiece::HSVClusterTrackerNodeTestPeer::Run(tracker, image, mask, pose));
      results.push_back(result);
      std::ofstream(argv[3]) << results.dump() << '\n';
    }
    std::ofstream(argv[3]) << results.dump() << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
