#include "gamepiece/ellipse.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// Keep the BOS test cases and calculations unchanged.
// NOLINTBEGIN(modernize-use-designated-initializers, modernize-use-std-numbers, modernize-use-trailing-return-type)
namespace {

using gamepiece::Ellipse;

struct EllipseVisualCase {
  std::string name;
  Ellipse first;
  Ellipse second;
  std::size_t expected_intersections;
};

auto EllipseValue(const Ellipse& ellipse, const cv::Point2d& point) -> double {
  const double cosine = std::cos(ellipse.rotation);
  const double sine = std::sin(ellipse.rotation);
  const cv::Point2d offset = point - ellipse.center;
  const double x = cosine * offset.x + sine * offset.y;
  const double y = -sine * offset.x + cosine * offset.y;
  return x * x / (ellipse.semi_axes.width * ellipse.semi_axes.width) +
         y * y / (ellipse.semi_axes.height * ellipse.semi_axes.height);
}

auto EllipseBoundary(const Ellipse& ellipse, double theta) -> cv::Point2d {
  const double cosine = std::cos(ellipse.rotation);
  const double sine = std::sin(ellipse.rotation);
  const double x = ellipse.semi_axes.width * std::cos(theta);
  const double y = ellipse.semi_axes.height * std::sin(theta);
  return {ellipse.center.x + cosine * x - sine * y,
          ellipse.center.y + sine * x + cosine * y};
}

auto RotatePoint(const cv::Point2d& point, double angle) -> cv::Point2d {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {cosine * point.x - sine * point.y, sine * point.x + cosine * point.y};
}

auto RotateEllipse(Ellipse ellipse, double angle) -> Ellipse {
  ellipse.center = RotatePoint(ellipse.center, angle);
  ellipse.rotation += angle;
  return ellipse;
}

auto IntersectionVisualCases() -> std::vector<EllipseVisualCase> {
  // The tangent cases include external tangency, internal tangency, and
  // rotated homothetic ellipses.  The three-intersection cases deliberately
  // combine one tangency with two ordinary crossings.
  const Ellipse tangent_external_a{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse tangent_external_b{{2.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse tangent_internal_a{{0.0, 0.0}, {3.0, 1.5}, 0.35};
  const Ellipse tangent_internal_b{
      {1.8 * std::cos(0.35), 1.8 * std::sin(0.35)}, {1.2, 0.6}, 0.35};
  const Ellipse tangent_rotated_a{{-0.4, 0.2}, {2.0, 1.0}, -0.45};
  const Ellipse tangent_rotated_b{
      {-0.4 + 1.2 * std::cos(-0.45), 0.2 + 1.2 * std::sin(-0.45)},
      {0.8, 0.4},
      -0.45};

  const Ellipse two_circle_a{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse two_circle_b{{1.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse two_ellipse_a{{-0.3, 0.1}, {2.0, 0.9}, 0.25};
  const Ellipse two_ellipse_b{{1.35, 0.25}, {1.1, 0.7}, -0.4};
  const Ellipse two_rotated_a{{0.0, 0.0}, {2.0, 1.0}, 0.6};
  const Ellipse two_rotated_b{{2.15, 0.15}, {1.3, 0.8}, -0.2};

  const Ellipse three_a{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse three_b{{0.0, 0.25}, {2.0, 0.75}, 0.0};
  const Ellipse three_scaled_a{{0.0, 0.0}, {1.3, 1.3}, 0.0};
  const Ellipse three_scaled_b{{0.0, 0.325}, {2.6, 0.975}, 0.0};
  const auto three_rotated_a = RotateEllipse(three_a, 0.52);
  const auto three_rotated_b = RotateEllipse(three_b, 0.52);

  const Ellipse four_a{{0.0, 0.0}, {2.0, 1.0}, 0.0};
  const Ellipse four_b{{0.0, 0.0}, {1.0, 2.0}, 0.0};
  const auto four_rotated_a = RotateEllipse(four_a, 0.37);
  const auto four_rotated_b = RotateEllipse(four_b, -0.29);
  const Ellipse four_offset_a{{-0.2, 0.1}, {2.0, 1.2}, 0.18};
  const Ellipse four_offset_b{{-0.05, 0.05}, {1.4, 1.8}, -0.32};

  return {
      {"1A external tangent", tangent_external_a, tangent_external_b, 1},
      {"1B internal tangent", tangent_internal_a, tangent_internal_b, 1},
      {"1C rotated tangent", tangent_rotated_a, tangent_rotated_b, 1},
      {"2A equal circles", two_circle_a, two_circle_b, 2},
      {"2B skew ellipses", two_ellipse_a, two_ellipse_b, 2},
      {"2C rotated pair", two_rotated_a, two_rotated_b, 2},
      {"3A tangent + crossings", three_a, three_b, 3},
      {"3B scaled tangent + crossings", three_scaled_a, three_scaled_b, 3},
      {"3C rotated tangent + crossings", three_rotated_a, three_rotated_b, 3},
      {"4A centered cross", four_a, four_b, 4},
      {"4B rotated cross", four_rotated_a, four_rotated_b, 4},
      {"4C offset cross", four_offset_a, four_offset_b, 4},
  };
}

auto DrawText(cv::Mat& image, const std::string& text, cv::Point origin,
              double scale = 0.47, const cv::Scalar& color = {25, 25, 25})
    -> void {
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1,
              cv::LINE_AA);
}

auto DrawEllipseContactSheet(
    const std::vector<EllipseVisualCase>& cases,
    const std::vector<std::vector<cv::Point2d>>& points,
    const std::vector<double>& overlap_areas, const std::string& output_path)
    -> void {
  constexpr int kTileWidth = 520;
  constexpr int kTileHeight = 390;
  constexpr int kColumns = 3;
  const int rows = static_cast<int>((cases.size() + kColumns - 1) / kColumns);
  cv::Mat sheet(rows * kTileHeight, kColumns * kTileWidth, CV_8UC3,
                cv::Scalar(248, 248, 244));

  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& test_case = cases[index];
    const int tile_x = static_cast<int>(index % kColumns) * kTileWidth;
    const int tile_y = static_cast<int>(index / kColumns) * kTileHeight;
    cv::Mat tile = sheet(cv::Rect(tile_x, tile_y, kTileWidth, kTileHeight));

    auto extent = [](const Ellipse& ellipse) {
      const double cosine = std::cos(ellipse.rotation);
      const double sine = std::sin(ellipse.rotation);
      return cv::Point2d(std::hypot(ellipse.semi_axes.width * cosine,
                                    ellipse.semi_axes.height * sine),
                         std::hypot(ellipse.semi_axes.width * sine,
                                    ellipse.semi_axes.height * cosine));
    };
    const auto first_extent = extent(test_case.first);
    const auto second_extent = extent(test_case.second);
    const double min_x = std::min(test_case.first.center.x - first_extent.x,
                                  test_case.second.center.x - second_extent.x);
    const double max_x = std::max(test_case.first.center.x + first_extent.x,
                                  test_case.second.center.x + second_extent.x);
    const double min_y = std::min(test_case.first.center.y - first_extent.y,
                                  test_case.second.center.y - second_extent.y);
    const double max_y = std::max(test_case.first.center.y + first_extent.y,
                                  test_case.second.center.y + second_extent.y);
    constexpr double kMargin = 48.0;
    const double scale =
        std::min((kTileWidth - 2.0 * kMargin) / (max_x - min_x),
                 (kTileHeight - 2.0 * kMargin) / (max_y - min_y));
    const cv::Point2d world_center{0.5 * (min_x + max_x),
                                   0.5 * (min_y + max_y)};
    const cv::Point2d pixel_center{kTileWidth * 0.5, kTileHeight * 0.55};
    auto to_pixel = [&](const cv::Point2d& point) {
      return cv::Point(
          static_cast<int>(
              std::lround(pixel_center.x + (point.x - world_center.x) * scale)),
          static_cast<int>(std::lround(pixel_center.y -
                                       (point.y - world_center.y) * scale)));
    };
    auto polygon = [&](const Ellipse& ellipse) {
      std::vector<cv::Point> result;
      result.reserve(361);
      for (int sample = 0; sample <= 360; ++sample) {
        result.push_back(to_pixel(
            EllipseBoundary(ellipse, 2.0 * std::numbers::pi * sample / 360.0)));
      }
      return result;
    };

    const auto first_polygon = polygon(test_case.first);
    const auto second_polygon = polygon(test_case.second);
    cv::Mat first_mask(tile.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat second_mask(tile.size(), CV_8UC1, cv::Scalar(0));
    cv::fillPoly(first_mask, std::vector<std::vector<cv::Point>>{first_polygon},
                 cv::Scalar(255));
    cv::fillPoly(second_mask,
                 std::vector<std::vector<cv::Point>>{second_polygon},
                 cv::Scalar(255));

    auto blend_mask = [&](const cv::Mat& mask, const cv::Scalar& color,
                          double alpha) {
      cv::Mat overlay(tile.size(), tile.type(), color);
      cv::Mat blended;
      cv::addWeighted(tile, 1.0 - alpha, overlay, alpha, 0.0, blended);
      blended.copyTo(tile, mask);
    };
    blend_mask(first_mask, cv::Scalar(220, 150, 50), 0.22);
    blend_mask(second_mask, cv::Scalar(90, 100, 225), 0.22);
    cv::Mat overlap_mask;
    cv::bitwise_and(first_mask, second_mask, overlap_mask);
    blend_mask(overlap_mask, cv::Scalar(65, 180, 75), 0.28);

    cv::polylines(tile, first_polygon, true, cv::Scalar(210, 105, 25), 2,
                  cv::LINE_AA);
    cv::polylines(tile, second_polygon, true, cv::Scalar(65, 70, 190), 2,
                  cv::LINE_AA);
    for (const auto& point : points[index]) {
      cv::drawMarker(tile, to_pixel(point), cv::Scalar(20, 20, 220),
                     cv::MARKER_CROSS, 15, 2, cv::LINE_AA);
      cv::circle(tile, to_pixel(point), 3, cv::Scalar(255, 255, 255), -1,
                 cv::LINE_AA);
    }

    const double first_area = std::numbers::pi *
                              test_case.first.semi_axes.width *
                              test_case.first.semi_axes.height;
    const double second_area = std::numbers::pi *
                               test_case.second.semi_axes.width *
                               test_case.second.semi_axes.height;
    DrawText(tile, test_case.name, {10, 22}, 0.55, {10, 10, 10});
    DrawText(tile,
             "intersections: " + std::to_string(points[index].size()) +
                 " (expected " +
                 std::to_string(cases[index].expected_intersections) + ")",
             {10, 43}, 0.43, {45, 45, 45});
    DrawText(tile, "ellipse 1 area: " + std::to_string(first_area), {10, 66},
             0.42, {210, 105, 25});
    DrawText(tile, "ellipse 2 area: " + std::to_string(second_area), {10, 86},
             0.42, {65, 70, 190});

    cv::Point overlap_label = to_pixel(
        {0.5 * (test_case.first.center.x + test_case.second.center.x),
         0.5 * (test_case.first.center.y + test_case.second.center.y)});
    const auto moments = cv::moments(overlap_mask, true);
    if (moments.m00 > 0.0) {
      overlap_label = {
          static_cast<int>(std::lround(moments.m10 / moments.m00)),
          static_cast<int>(std::lround(moments.m01 / moments.m00))};
    }
    const std::string overlap_text =
        "predicted overlap: " + std::to_string(overlap_areas[index]);
    DrawText(tile, overlap_text,
             {std::clamp(overlap_label.x - 75, 8, kTileWidth - 235),
              std::clamp(overlap_label.y, 112, kTileHeight - 12)},
             0.42, {15, 110, 15});
  }

  ASSERT_TRUE(cv::imwrite(output_path, sheet))
      << "Could not write ellipse visualization to " << output_path;
}

TEST(EllipseTest, FindsCircleIntersectionsAndOverlapArea) {
  const Ellipse first{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse second{{1.0, 0.0}, {1.0, 1.0}, 0.0};

  const auto intersections = gamepiece::ellipse_intersections(first, second);
  ASSERT_EQ(intersections.size(), 2U);
  for (const auto& point : intersections) {
    EXPECT_NEAR(point.x, 0.5, 1e-7);
    EXPECT_NEAR(std::abs(point.y), std::sqrt(3.0) / 2.0, 1e-7);
  }

  const double expected_area = 2.0 * std::acos(0.5) - std::sqrt(3.0) / 2.0;
  EXPECT_NEAR(gamepiece::ellipse_overlap_area(first, second), expected_area,
              1e-7);
}

TEST(EllipseTest, HandlesDisjointAndContainedEllipses) {
  const Ellipse large{{0.0, 0.0}, {2.0, 2.0}, 0.0};
  const Ellipse contained{{0.2, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse disjoint{{5.0, 0.0}, {1.0, 0.5}, 0.3};

  EXPECT_DOUBLE_EQ(gamepiece::ellipse_overlap_area(large, disjoint), 0.0);
  EXPECT_NEAR(gamepiece::ellipse_overlap_area(large, contained),
              std::numbers::pi, 1e-8);
}

TEST(EllipseTest, HandlesTangencyAndFourIntersections) {
  const Ellipse circle{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  const Ellipse tangent{{2.0, 0.0}, {1.0, 1.0}, 0.0};
  EXPECT_EQ(gamepiece::ellipse_intersections(circle, tangent).size(), 1U);
  EXPECT_NEAR(gamepiece::ellipse_overlap_area(circle, tangent), 0.0, 1e-10);

  const Ellipse horizontal{{0.0, 0.0}, {2.0, 1.0}, 0.0};
  const Ellipse vertical{{0.0, 0.0}, {1.0, 2.0}, 0.0};
  const auto intersections =
      gamepiece::ellipse_intersections(horizontal, vertical);
  ASSERT_EQ(intersections.size(), 4U);
  for (const auto& point : intersections) {
    EXPECT_NEAR(std::abs(point.x), std::sqrt(0.8), 1e-7);
    EXPECT_NEAR(std::abs(point.y), std::sqrt(0.8), 1e-7);
  }
  EXPECT_NEAR(gamepiece::ellipse_overlap_area(horizontal, vertical),
              4.0 * std::numbers::pi, 1e-8);
}

TEST(EllipseTest, RejectsInvalidRadii) {
  const Ellipse invalid{{0.0, 0.0}, {0.0, 1.0}, 0.0};
  const Ellipse valid{{0.0, 0.0}, {1.0, 1.0}, 0.0};
  EXPECT_THROW(gamepiece::ellipse_intersections(invalid, valid),
               std::invalid_argument);
}

TEST(EllipseTest, VisualizesThreeCasesForEveryIntersectionCount) {
  const auto cases = IntersectionVisualCases();
  std::vector<std::vector<cv::Point2d>> all_intersections;
  std::vector<double> all_overlap_areas;
  all_intersections.reserve(cases.size());
  all_overlap_areas.reserve(cases.size());

  for (const auto& test_case : cases) {
    const auto intersections =
        gamepiece::ellipse_intersections(test_case.first, test_case.second);
    EXPECT_EQ(intersections.size(), test_case.expected_intersections)
        << test_case.name;
    for (const auto& point : intersections) {
      EXPECT_NEAR(EllipseValue(test_case.first, point), 1.0, 1e-5)
          << test_case.name << " first ellipse at " << point;
      EXPECT_NEAR(EllipseValue(test_case.second, point), 1.0, 1e-5)
          << test_case.name << " second ellipse at " << point;
    }

    const double overlap =
        gamepiece::ellipse_overlap_area(test_case.first, test_case.second);
    EXPECT_TRUE(std::isfinite(overlap)) << test_case.name;
    const double first_area = std::numbers::pi *
                              test_case.first.semi_axes.width *
                              test_case.first.semi_axes.height;
    const double second_area = std::numbers::pi *
                               test_case.second.semi_axes.width *
                               test_case.second.semi_axes.height;
    if (test_case.expected_intersections <= 1) {
      const bool first_is_inner = first_area < second_area;
      const Ellipse& inner =
          first_is_inner ? test_case.first : test_case.second;
      const Ellipse& outer =
          first_is_inner ? test_case.second : test_case.first;
      const bool contained = EllipseValue(outer, inner.center) <= 1.0 + 1e-8;
      EXPECT_NEAR(overlap, contained ? std::min(first_area, second_area) : 0.0,
                  1e-10)
          << test_case.name;
    } else if (test_case.expected_intersections > 2) {
      EXPECT_NEAR(overlap, first_area + second_area, 1e-8) << test_case.name;
    } else {
      EXPECT_GT(overlap, 0.0) << test_case.name;
      EXPECT_LT(overlap, std::min(first_area, second_area)) << test_case.name;
    }
    all_intersections.push_back(intersections);
    all_overlap_areas.push_back(overlap);
  }

  const char* configured_path = std::getenv("ELLIPSE_VISUAL_OUTPUT");
  const std::string output_path =
      configured_path == nullptr ? "/tmp/ellipse_intersection_visualization.png"
                                 : configured_path;
  DrawEllipseContactSheet(cases, all_intersections, all_overlap_areas,
                          output_path);
}

}  // namespace
// NOLINTEND(modernize-use-designated-initializers, modernize-use-std-numbers, modernize-use-trailing-return-type)
