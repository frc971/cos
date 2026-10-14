#include "gamepiece/ellipse.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <unsupported/Eigen/Polynomials>

namespace gamepiece {
namespace {

constexpr double kGeometryTolerance = 1e-8;

struct EllipseData {
  cv::Vec2d center;
  cv::Vec2d semi_major_axis_vector;
  cv::Vec2d semi_minor_axis_vector;
  cv::Matx22d test_on_ellipse;
  cv::Matx22d rotation;
  double area, a, b;
};

auto GetEllipseData(const Ellipse& ellipse) -> EllipseData {
  const double a = ellipse.semi_axes.width;
  const double b = ellipse.semi_axes.height;
  if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b) ||
      !std::isfinite(ellipse.center.x) || !std::isfinite(ellipse.center.y) ||
      !std::isfinite(ellipse.rotation)) {
    throw std::invalid_argument(
        "Ellipse values must be finite and radii positive");
  }

  const double cosine = std::cos(ellipse.rotation);
  const double sine = std::sin(ellipse.rotation);
  const cv::Matx22d rotation(cosine, -sine, sine, cosine);

  EllipseData result;
  result.center = {ellipse.center.x, ellipse.center.y};
  result.semi_major_axis_vector = cv::Vec2d(cosine * a, sine * a);
  result.semi_minor_axis_vector = cv::Vec2d(-sine * b, cosine * b);
  result.test_on_ellipse = rotation *
                           cv::Matx22d(1.0 / (a * a), 0.0, 0.0, 1.0 / (b * b)) *
                           rotation.t();
  result.rotation = rotation;
  result.area = std::numbers::pi * a * b;
  result.a = a;
  result.b = b;
  return result;
}

auto NormalizedDistanceSquared(const cv::Vec2d& point,
                               const EllipseData& ellipse) -> double {
  const cv::Vec2d offset = point - ellipse.center;
  return offset.dot(ellipse.test_on_ellipse * offset);
}

auto SameEllipse(const EllipseData& first, const EllipseData& second) -> bool {
  const double center_scale =
      1.0 + cv::norm(first.center) + cv::norm(second.center);
  const double matrix_scale =
      1.0 + cv::norm(first.test_on_ellipse) + cv::norm(second.test_on_ellipse);
  return cv::norm(first.center - second.center) <=
             kGeometryTolerance * center_scale &&
         cv::norm(first.test_on_ellipse - second.test_on_ellipse) <=
             kGeometryTolerance * matrix_scale;
}

auto QuarticCoefficients(const EllipseData& source_ellipse,
                         const EllipseData& constraint_ellipse,
                         double source_rotation) -> cv::Vec<double, 5> {
  const cv::Vec2d phase_radial =
      source_ellipse.semi_major_axis_vector * std::cos(source_rotation) +
      source_ellipse.semi_minor_axis_vector * std::sin(source_rotation);

  const cv::Vec2d phase_tangent =
      -source_ellipse.semi_major_axis_vector * std::sin(source_rotation) +
      source_ellipse.semi_minor_axis_vector * std::cos(source_rotation);

  const cv::Vec2d center_offset =
      source_ellipse.center - constraint_ellipse.center;

  // polynomial a bu cu^2 order
  const std::array<cv::Vec2d, 3> numerator_coefficients{
      center_offset + phase_radial,
      2.0 * phase_tangent,
      center_offset - phase_radial,
  };

  cv::Vec<double, 5> quartic = cv::Vec<double, 5>::all(0.0);

  for (int i = 0; i <= 2; ++i) {
    for (int j = 0; j <= 2; ++j) {
      quartic[i + j] += numerator_coefficients[i].dot(
          constraint_ellipse.test_on_ellipse * numerator_coefficients[j]);
    }
  }

  // from denominator (1 + u^2)^2
  quartic[0] -= 1.0;
  quartic[2] -= 2.0;
  quartic[4] -= 1.0;

  return quartic;
}

auto PointAt(const EllipseData& ellipse, double theta) -> cv::Vec2d {
  return ellipse.center + ellipse.semi_major_axis_vector * std::cos(theta) +
         ellipse.semi_minor_axis_vector * std::sin(theta);
}

auto ThetaFromPoint(const EllipseData& ellipse, const cv::Point2d& point)
    -> double {
  const cv::Vec2d centered =
      ellipse.rotation.t() * (cv::Vec2d(point.x, point.y) - ellipse.center);

  return std::atan2(centered[1] / ellipse.b, centered[0] / ellipse.a);
}

auto SectorArea(const EllipseData& ellipse, double angle) -> double {
  return ellipse.a * ellipse.b * 0.5 * std::abs(angle);
}

auto CurvedArea(const EllipseData& ellipse_1, const EllipseData& ellipse_2,
                const std::vector<cv::Point2d>& intersections) -> double {
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  const double lower_bound = ThetaFromPoint(ellipse_1, intersections[0]);
  const double upper_bound = ThetaFromPoint(ellipse_1, intersections[1]);
  double counterclockwise_angle = upper_bound - lower_bound;
  if (counterclockwise_angle < 0.0) {
    counterclockwise_angle += kTwoPi;
  }

  const double midpoint_angle = lower_bound + 0.5 * counterclockwise_angle;
  const bool counterclockwise_arc_is_inside =
      NormalizedDistanceSquared(PointAt(ellipse_1, midpoint_angle),
                                ellipse_2) <= 1.0 + kGeometryTolerance;
  const double angle = counterclockwise_arc_is_inside
                           ? counterclockwise_angle
                           : kTwoPi - counterclockwise_angle;
  double ellipse_1_curved_area =
      SectorArea(ellipse_1, angle) -
      0.5 * ellipse_1.a * ellipse_1.b * std::sin(angle);
  return ellipse_1_curved_area;
}

}  // namespace

auto ellipse_intersections(const Ellipse& first, const Ellipse& second)
    -> std::vector<cv::Point2d> {
  const EllipseData first_data = GetEllipseData(first);
  const EllipseData second_data = GetEllipseData(second);
  if (SameEllipse(first_data, second_data)) {
    return {};
  }

  // try different phases because there is a singularity at phi - theta = pi which makes
  // the polynomial unsolvable. Indication that the result is solvable is that the
  // u4 coefficient is reasonably large instead of collapsing to 0
  constexpr std::array<double, 8> phases{0.0,  0.37, 0.79, 1.21,
                                         1.63, 2.05, 2.47, 2.89};
  cv::Vec<double, 5> coefficients;
  double accepted_phase = -1;
  for (const double candidate_phase : phases) {
    const cv::Vec<double, 5> candidate =
        QuarticCoefficients(first_data, second_data, candidate_phase);
    const bool well_conditioned =
        std::abs(candidate[4]) / cv::norm(candidate) > 1e-8;
    if (well_conditioned) {
      accepted_phase = candidate_phase;
      coefficients = candidate;
    }
  }

  if (accepted_phase == -1) {
    LOG(WARNING)
        << "All 8 phase attempts failed to produce a solvable polynomial";
    return {};
  }

  double largest_coefficient = 0.0;
  for (const double coefficient : coefficients.val) {
    largest_coefficient = std::max(largest_coefficient, std::abs(coefficient));
  }
  coefficients /= largest_coefficient;

  std::array<std::complex<double>, 4> roots;
  Eigen::Matrix<double, 5, 1> eigen_coefficients;
  for (int i = 0; i < 5; ++i) {
    eigen_coefficients[i] = coefficients[i];
  }
  const Eigen::PolynomialSolver<double, 4> solver(eigen_coefficients);
  const auto& eigen_roots = solver.roots();
  std::ranges::copy(eigen_roots, roots.begin());

  std::vector<cv::Point2d> intersections;
  for (const std::complex<double>& root : roots) {
    if (std::abs(root.imag()) > 1e-6 * (1.0 + std::abs(root.real()))) {
      continue;
    }
    const double theta = accepted_phase + 2.0 * std::atan(root.real());
    const cv::Vec2d point = PointAt(first_data, theta);
    const double residual =
        std::abs(NormalizedDistanceSquared(point, second_data) - 1.0);
    if (residual > 1e-6) {
      LOG(WARNING) << "Real eigen root didn't actually lie on the ellipse";
      continue;
    }

    const cv::Point2d result(point[0], point[1]);
    const double scale = 1.0 + cv::norm(point);
    const bool duplicate =
        std::ranges::any_of(intersections, [&](const cv::Point2d& old) -> bool {
          return cv::norm(result - old) <= 1e-6 * scale;
        });
    if (!duplicate) {
      intersections.push_back(result);
    }
  }
  return intersections;
}

auto ellipse_overlap_area(const Ellipse& first, const Ellipse& second)
    -> double {
  const EllipseData ellipse_1 = GetEllipseData(first);
  const EllipseData ellipse_2 = GetEllipseData(second);
  if (SameEllipse(ellipse_1, ellipse_2)) {
    return ellipse_1.area;
  }

  const std::vector<cv::Point2d> intersections =
      ellipse_intersections(first, second);
  if (intersections.size() <= 1) {
    const bool ellipse_1_inner = ellipse_1.area < ellipse_2.area;
    const EllipseData& possibly_inner_ellipse =
        ellipse_1_inner ? ellipse_1 : ellipse_2;
    const EllipseData& possibly_outer_ellipse =
        ellipse_1_inner ? ellipse_2 : ellipse_1;
    const bool contained =
        NormalizedDistanceSquared(possibly_inner_ellipse.center,
                                  possibly_outer_ellipse) <=
        1.0 + kGeometryTolerance;
    return contained ? possibly_inner_ellipse.area : 0.0;
  } else if (intersections.size() > 2) {
    // not meant to be accurate, if there are 2+ intersection points the clusters
    // should be merged anyway
    return ellipse_1.area + ellipse_2.area;
  }

  double area = CurvedArea(ellipse_1, ellipse_2, intersections) +
                CurvedArea(ellipse_2, ellipse_1, intersections);
  return area;
}

}  // namespace gamepiece
