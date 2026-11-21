// Cache raw per-frame robot poses (meters, radians); null means no estimate.
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <frc/apriltag/AprilTagDetector.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "absl/log/check.h"
#include "camera/camera_config.h"
#include "localization/multi_tag_solver_node.h"

auto main(int argc, char** argv) -> int {
  if (argc < 5) {
    std::cerr << "Usage: robot_positions <intrinsics.json> <extrinsics.json> "
                 "<poses.json> <frame.jpg>...\n";
    return 1;
  }
  try {
    localization::MultiTagSolverNode solver(
        "tags", "pose", camera::Intrinsics{argv[1]}, camera::Extrinsics{argv[2]});
    frc::AprilTagDetector detector;
    CHECK(detector.AddFamily("tag36h11"));
    frc::AprilTagDetector::Config config;
    config.quadDecimate = 1.0F;
    detector.SetConfig(config);
    auto poses = nlohmann::json::object();
    for (int i = 4; i < argc; ++i) {
      const cv::Mat frame = cv::imread(argv[i], cv::IMREAD_COLOR);
      CHECK(!frame.empty()) << "Cannot decode frame: " << argv[i];
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      const auto name = std::filesystem::path(argv[i]).filename().string();
      CHECK(!poses.contains(name)) << "Duplicate frame name: " << name;
      std::vector<localization::tag_detection_t> detections;
      for (const auto* tag : detector.Detect(
               gray.cols, gray.rows, static_cast<int>(gray.step), gray.data)) {
        localization::tag_detection_t detection{
            .tag_id = tag->GetId(), .corners = {}};
        for (int corner = 0; corner < 4; ++corner) {
          detection.corners[corner] = {tag->GetCorner(corner).x,
                                       tag->GetCorner(corner).y};
        }
        detections.push_back(detection);
      }
      poses[name] = nullptr;
      if (const auto estimate = solver.AmbiguousSolve(detections)) {
        const auto pose = estimate->pos1.pose.ToPose2d();
        poses[name] = {{"x", pose.X().value()}, {"y", pose.Y().value()},
                       {"yaw", pose.Rotation().Radians().value()}};
      }
    }
    std::ofstream output(argv[3]);
    CHECK(output.is_open()) << "Cannot write poses: " << argv[3];
    output << poses.dump(2) << '\n';
    CHECK(output.good());
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
