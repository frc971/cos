#include "pathing/dijkstra_spline_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {
auto Pose(double x, double y, double heading = 0) -> frc::Pose2d {
  return {units::meter_t{x}, units::meter_t{y},
          frc::Rotation2d{units::degree_t{heading}}};
}

auto Context() -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1);
}

class DijkstraSplineNodeTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("spline_test_" + std::to_string(timestamp) + ".json");
    WriteGrid({"......", "......", "......", "......", "......", "......"});
  }
  void TearDown() override { std::filesystem::remove(path_); }

  void WriteGrid(const std::vector<std::string>& rows, double size_x = 0,
                 double size_y = 0) {
    rows_ = rows;
    size_x_ = size_x == 0 ? rows.front().size() : size_x;
    size_y_ = size_y == 0 ? rows.size() : size_y;
    std::vector<std::vector<bool>> grid;
    for (const auto& row : rows) {
      auto& cells = grid.emplace_back();
      for (char c : row)
        cells.push_back(c == '#');
    }
    std::ofstream file(path_);
    file << nlohmann::json{{"field_size", {{"x", size_x_}, {"y", size_y_}}},
                           {"nodeSizeMeters", 1.0},
                           {"grid", grid}};
  }

  auto Plan(frc::Pose2d start, frc::Pose2d goal,
            pathing::SplineSettings settings = {})
      -> std::shared_ptr<pathing::SplinePathMessage> {
    pathing::DijkstraSplineNode node("pose", "goal", "spline", path_, settings);
    auto context = Context();
    context->SetMessage("pose", std::make_shared<pathing::PoseMessage>(start));
    context->SetMessage("goal", std::make_shared<pathing::PoseMessage>(goal));
    node.CreateCallback()(context);
    return context->GetSharedMessage<pathing::SplinePathMessage>("spline");
  }

  // Independent dense checks against the fixture's rectangles. Production
  // clearance uses recursive control hulls, not these sampled point checks.
  void ExpectClear(const pathing::SplinePathMessage& output) {
    ASSERT_EQ(output.status, pathing::SplineStatus::Ready)
        << "refinements=" << output.refinement_passes
        << ", waypoints=" << output.waypoints.size();
    for (const auto& spline : output.splines) {
      const auto& a = spline.GetInitialControlVector();
      const auto& b = spline.GetFinalControlVector();
      for (int i = 0; i <= 2000; ++i) {
        const double t = i / 2000.0, u = 1 - t;
        const double h0 = u * u * (1 + 2 * t);
        const double h1 = t * u * u;
        const double h2 = t * t * (3 - 2 * t);
        const double h3 = -t * t * u;
        const double x = h0 * a.x[0] + h1 * a.x[1] + h2 * b.x[0] + h3 * b.x[1];
        const double y = h0 * a.y[0] + h1 * a.y[1] + h2 * b.y[0] + h3 * b.y[1];
        ASSERT_GE(x, 0);
        ASSERT_GE(y, 0);
        ASSERT_LT(x, size_x_);
        ASSERT_LT(y, size_y_);
        for (size_t row = 0; row < rows_.size(); ++row) {
          for (size_t col = 0; col < rows_[row].size(); ++col) {
            if (rows_[row][col] == '#') {
              ASSERT_FALSE(x >= col && x <= col + 1 && y >= row && y <= row + 1)
                  << "Curve intersects blocked cell " << col << "," << row;
            }
          }
        }
      }
    }
  }

  std::filesystem::path path_;
  std::vector<std::string> rows_;
  double size_x_ = 0;
  double size_y_ = 0;
};

TEST_F(DijkstraSplineNodeTest,
       StraightSplinePreservesEndpointsAndSampleSpacing) {
  auto result = Plan(Pose(0.2, 0.4), Pose(5.7, 4.3));
  ASSERT_NE(result, nullptr);
  ExpectClear(*result);
  ASSERT_EQ(result->splines.size(), 1);
  ASSERT_GE(result->samples.size(), 2);
  EXPECT_DOUBLE_EQ(result->samples.front().X().value(), 0.2);
  EXPECT_DOUBLE_EQ(result->samples.front().Y().value(), 0.4);
  EXPECT_DOUBLE_EQ(result->samples.back().X().value(), 5.7);
  EXPECT_DOUBLE_EQ(result->samples.back().Y().value(), 4.3);
  for (size_t i = 1; i < result->samples.size(); ++i)
    EXPECT_LE(result->samples[i].Distance(result->samples[i - 1]).value(),
              0.02 + 1e-12);
  for (size_t i = 1; i < result->raw_route.size(); ++i)
    EXPECT_GT(result->raw_route[i].Distance(result->raw_route[i - 1]).value(),
              0);
}

TEST_F(DijkstraSplineNodeTest, IgnoresRobotHeadingWhenChoosingTangents) {
  auto a = Plan(Pose(0.5, 0.5, 90), Pose(5.5, 0.5, 180));
  auto b = Plan(Pose(0.5, 0.5, -90), Pose(5.5, 0.5, 0));
  ASSERT_EQ(a->status, pathing::SplineStatus::Ready);
  ASSERT_EQ(b->status, pathing::SplineStatus::Ready);
  ASSERT_EQ(a->splines.size(), 1);
  EXPECT_TRUE(a->splines.front().Coefficients().isApprox(
      b->splines.front().Coefficients()));
  EXPECT_DOUBLE_EQ(a->splines.front().GetInitialControlVector().y[1], 0);
}

TEST_F(DijkstraSplineNodeTest, AddsControlPointsToResolveCornerCollision) {
  WriteGrid({".......", ".#####.", ".#####.", ".#...#.", "...#..."});
  pathing::SplineSettings initial_settings;
  initial_settings.turn_penalty = 1;
  initial_settings.max_refinement_passes = 0;
  auto initial = Plan(Pose(0.5, 3.5), Pose(6.5, 3.5), initial_settings);
  EXPECT_EQ(initial->status, pathing::SplineStatus::FitFailed);
  EXPECT_TRUE(initial->splines.empty());
  EXPECT_TRUE(initial->samples.empty());
  auto settings = initial_settings;
  settings.max_refinement_passes = 32;
  auto result = Plan(Pose(0.5, 3.5), Pose(6.5, 3.5), settings);
  ExpectClear(*result);
  EXPECT_GT(result->refinement_passes, 0);
  EXPECT_GT(result->waypoints.size(), initial->waypoints.size());
  for (size_t i = 1; i < result->splines.size(); ++i) {
    const auto& prev = result->splines[i - 1];
    const auto& next = result->splines[i];
    const auto& a = prev.GetFinalControlVector();
    const auto& b = next.GetInitialControlVector();
    const double h0 =
        result->waypoints[i].Distance(result->waypoints[i - 1]).value();
    const double h1 =
        result->waypoints[i + 1].Distance(result->waypoints[i]).value();
    EXPECT_NEAR(a.x[0], b.x[0], 1e-9);
    EXPECT_NEAR(a.y[0], b.y[0], 1e-9);
    EXPECT_NEAR(a.x[1] / h0, b.x[1] / h1, 1e-9);
    EXPECT_NEAR(a.y[1] / h0, b.y[1] / h1, 1e-9);
    const auto c0 = prev.Coefficients();
    const auto c1 = next.Coefficients();
    EXPECT_NEAR(c0.row(4).sum() / (h0 * h0), c1(4, 1) / (h1 * h1), 1e-8);
    EXPECT_NEAR(c0.row(5).sum() / (h0 * h0), c1(5, 1) / (h1 * h1), 1e-8);
  }
}

TEST_F(DijkstraSplineNodeTest,
       ControlPointBudgetFailureRetainsOnlyRawGeometry) {
  WriteGrid({".......", ".#####.", ".#####.", ".#...#.", "...#..."});
  pathing::SplineSettings settings;
  settings.max_control_points = 2;
  auto result = Plan(Pose(0.5, 3.5), Pose(6.5, 3.5), settings);
  EXPECT_EQ(result->status, pathing::SplineStatus::FitFailed);
  EXPECT_FALSE(result->raw_route.empty());
  EXPECT_TRUE(result->splines.empty());
  EXPECT_TRUE(result->samples.empty());
}

TEST_F(DijkstraSplineNodeTest, CoincidentPositionsProduceSinglePoint) {
  auto result = Plan(Pose(0.5, 0.5), Pose(0.5, 0.5, 90));
  EXPECT_EQ(result->status, pathing::SplineStatus::Ready);
  EXPECT_TRUE(result->splines.empty());
  ASSERT_EQ(result->samples.size(), 1);
  EXPECT_DOUBLE_EQ(result->samples.front().X().value(), 0.5);
}

TEST_F(DijkstraSplineNodeTest, SameCellDistinctPositionsProduceDirectCurve) {
  auto result = Plan(Pose(0.1, 0.2), Pose(0.8, 0.7));
  ExpectClear(*result);
  EXPECT_EQ(result->raw_route.size(), 2);
  EXPECT_EQ(result->splines.size(), 1);
}

TEST_F(DijkstraSplineNodeTest, HandlesPartialEdgeCellsAndFieldBounds) {
  WriteGrid({"...", "..."}, 2.2, 1.3);
  auto result = Plan(Pose(0.05, 0.05), Pose(2.19, 1.29));
  ExpectClear(*result);
  EXPECT_EQ(Plan(Pose(0.5, 0.5), Pose(2.2, 1.29))->status,
            pathing::SplineStatus::InvalidInput);
}

TEST_F(DijkstraSplineNodeTest, ReportsInvalidAndUnreachableGoals) {
  WriteGrid({"..#...", "..#...", "..#...", "..#..."});
  EXPECT_EQ(Plan(Pose(0.5, 0.5), Pose(5.5, 0.5))->status,
            pathing::SplineStatus::NoPath);
  EXPECT_EQ(Plan(Pose(0.5, 0.5), Pose(2.5, 0.5))->status,
            pathing::SplineStatus::InvalidInput);
  EXPECT_EQ(Plan(Pose(-0.1, 0.5), Pose(1.5, 0.5))->status,
            pathing::SplineStatus::InvalidInput);
  EXPECT_EQ(
      Plan(Pose(std::numeric_limits<double>::quiet_NaN(), 0.5), Pose(1.5, 0.5))
          ->status,
      pathing::SplineStatus::InvalidInput);
}

TEST_F(DijkstraSplineNodeTest, DoesNotCutOrTouchBlockedCorners) {
  WriteGrid({".#", "#."});
  EXPECT_EQ(Plan(Pose(0.5, 0.5), Pose(1.5, 1.5))->status,
            pathing::SplineStatus::NoPath);
  WriteGrid({"#..", "..."});
  const auto result = Plan(Pose(1.0, 0.5), Pose(2.5, 0.5));
  EXPECT_EQ(result->status, pathing::SplineStatus::FitFailed);
  EXPECT_TRUE(result->splines.empty());
  EXPECT_TRUE(result->samples.empty());
}

TEST_F(DijkstraSplineNodeTest, SplinePreservesRobotRadiusAroundObstacles) {
  WriteGrid({"......", "......", "..#...", "......", "......", "......"});
  pathing::SplineSettings settings;
  settings.robot_radius_meters = 0.38;
  EXPECT_EQ(Plan(Pose(1.7, 2.5), Pose(4.5, 2.5), settings)->status,
            pathing::SplineStatus::InvalidInput);
  const auto result = Plan(Pose(0.5, 2.5), Pose(4.5, 2.5), settings);
  ExpectClear(*result);
  for (const auto& sample : result->samples) {
    const double x = sample.X().value(), y = sample.Y().value();
    const double dx = std::max({2.0 - x, x - 3.0, 0.0});
    const double dy = std::max({2.0 - y, y - 3.0, 0.0});
    EXPECT_GT(std::hypot(dx, dy), 0.38);
    EXPECT_GE(x, 0.38);
    EXPECT_GE(y, 0.38);
    EXPECT_LT(x + 0.38, 6.0);
    EXPECT_LT(y + 0.38, 6.0);
  }
}

TEST_F(DijkstraSplineNodeTest, StreamerSkipsMissingAndNullLayers) {
  const auto image_path = path_.string() + ".png";
  ASSERT_TRUE(cv::imwrite(image_path, cv::Mat::zeros(100, 100, CV_8UC3)));
  {
    streamer::NavgridStreamerNode streamer("/navgrid", 0, image_path);
    streamer.AddNavgrid("grid");
    streamer.AddPolyline("curve");
    int notifications = 0;
    streamer.RegisterCallback([&](const auto&) { ++notifications; });
    auto draw = streamer.CreateCallback();
    draw(Context());
    auto empty = Context();
    empty->SetMessage("grid", nullptr);
    empty->SetMessage("curve", nullptr);
    draw(empty);
    auto curve = Context();
    auto line = std::make_shared<streamer::PolylineMessage>();
    line->field_size_x = line->field_size_y = 6;
    line->points = {{0.5, 0.5}, {5.5, 5.5}};
    curve->SetMessage("grid", nullptr);
    curve->SetMessage("curve", line);
    draw(curve);
    EXPECT_EQ(notifications, 3);
  }
  std::filesystem::remove(image_path);
}

TEST_F(DijkstraSplineNodeTest, WaitsForDependenciesAndPublishesOncePerContext) {
  pathing::DijkstraSplineNode node("pose", "goal", "spline", path_);
  node.EnableSplineDebug("debug");
  EXPECT_EQ(node.GetDependencies().size(), 2);
  EXPECT_EQ(node.GetPublications().size(), 2);
  int notifications = 0;
  node.RegisterCallback([&](const auto&) { ++notifications; });
  auto callback = node.CreateCallback();
  auto context = Context();
  context->SetMessage("pose", nullptr);
  callback(context);
  EXPECT_FALSE(context->Exists("spline"));
  context->SetMessage("goal",
                      std::make_shared<pathing::PoseMessage>(Pose(1, 1)));
  callback(context);
  callback(context);
  EXPECT_EQ(notifications, 1);
  auto result = context->GetMessage<pathing::SplinePathMessage>("spline");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->status, pathing::SplineStatus::InvalidInput);
  auto debug = context->GetMessage<streamer::PolylineMessage>("debug");
  ASSERT_NE(debug, nullptr);
  EXPECT_TRUE(debug->points.empty());
}

TEST_F(DijkstraSplineNodeTest, PublishesPolylineAndValidatesChannels) {
  pathing::DijkstraSplineNode node("pose", "goal", "spline", path_);
  node.EnableSplineDebug("debug");
  EXPECT_NO_THROW(node.EnableSplineDebug("debug"));
  EXPECT_THROW(node.EnableSplineDebug("another"), std::logic_error);
  EXPECT_THROW(node.EnableSplineDebug("pose"), std::invalid_argument);
  auto context = Context();
  context->SetMessage("pose",
                      std::make_shared<pathing::PoseMessage>(Pose(0.5, 0.5)));
  context->SetMessage("goal",
                      std::make_shared<pathing::PoseMessage>(Pose(5.5, 5.5)));
  node.CreateCallback()(context);
  const auto* curve = context->GetMessage<pathing::SplinePathMessage>("spline");
  const auto* debug = context->GetMessage<streamer::PolylineMessage>("debug");
  ASSERT_NE(curve, nullptr);
  ASSERT_NE(debug, nullptr);
  EXPECT_EQ(debug->points.size(), curve->samples.size());
  EXPECT_DOUBLE_EQ(debug->field_size_x, 6);
  EXPECT_THROW((pathing::DijkstraSplineNode{"pose", "pose", "spline", path_}),
               std::invalid_argument);
}

TEST_F(DijkstraSplineNodeTest, RejectsInvalidSettings) {
  pathing::SplineSettings settings;
  settings.max_control_points = 1;
  EXPECT_THROW(
      (pathing::DijkstraSplineNode{"pose", "goal", "spline", path_, settings}),
      std::invalid_argument);
  settings = {};
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    settings.sample_spacing = units::meter_t{invalid};
    EXPECT_THROW((pathing::DijkstraSplineNode{"pose", "goal", "spline", path_,
                                              settings}),
                 std::invalid_argument);
  }
  settings = {};
  settings.turn_penalty = -1;
  EXPECT_THROW(
      (pathing::DijkstraSplineNode{"pose", "goal", "spline", path_, settings}),
      std::invalid_argument);
}

}  // namespace
