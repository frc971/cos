#include "pathing/dijkstra_spline_node.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "streamer/navgrid_streamer_node.h"

namespace pathing {
namespace {
using Bezier = std::array<cv::Point2d, 4>;
constexpr int kCollisionDepth = 16;
constexpr size_t kMaxSamples = 100000;

struct Knot {
  frc::Translation2d point;
  double along;
};

auto Point(const frc::Translation2d& point) -> cv::Point2d {
  return {point.X().value(), point.Y().value()};
}

auto Translation(cv::Point2d point) -> frc::Translation2d {
  return {units::meter_t{point.x}, units::meter_t{point.y}};
}

auto Controls(const frc::CubicHermiteSpline& spline) -> Bezier {
  const auto& a = spline.GetInitialControlVector();
  const auto& b = spline.GetFinalControlVector();
  return {cv::Point2d{a.x[0], a.y[0]},
          cv::Point2d{a.x[0] + a.x[1] / 3, a.y[0] + a.y[1] / 3},
          cv::Point2d{b.x[0] - b.x[1] / 3, b.y[0] - b.y[1] / 3},
          cv::Point2d{b.x[0], b.y[0]}};
}

auto Split(const Bezier& b) -> std::pair<Bezier, Bezier> {
  const auto p01 = (b[0] + b[1]) * 0.5;
  const auto p12 = (b[1] + b[2]) * 0.5;
  const auto p23 = (b[2] + b[3]) * 0.5;
  const auto p012 = (p01 + p12) * 0.5;
  const auto p123 = (p12 + p23) * 0.5;
  const auto middle = (p012 + p123) * 0.5;
  return {Bezier{b[0], p01, p012, middle}, Bezier{middle, p123, p23, b[3]}};
}

// The entire curve lies inside its control hull. A free bounding box certifies
// clearance; ambiguous hulls must be subdivided, never accepted by sampling.
auto IsClear(const DijkstraPlanner& planner, const Bezier& b,
             std::span<const DijkstraPlanner::Cell> cleared, int depth = 0)
    -> bool {
  double min_x = b[0].x, max_x = b[0].x;
  double min_y = b[0].y, max_y = b[0].y;
  for (const auto& p : b) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      return false;
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }
  if (planner.IsBoxFree(min_x, min_y, max_x, max_y, cleared))
    return true;
  if (depth == kCollisionDepth)
    return false;
  const auto [left, right] = Split(b);
  return IsClear(planner, left, cleared, depth + 1) &&
         IsClear(planner, right, cleared, depth + 1);
}

// Control-polygon length bounds arc length, so emitted points are at most the
// requested arc distance apart. Keep independent depth and allocation bounds.
auto Sample(const Bezier& b, double spacing,
            std::vector<frc::Translation2d>& samples, int depth = 0) -> bool {
  double length = 0;
  for (size_t i = 1; i < b.size(); ++i)
    length += cv::norm(b[i] - b[i - 1]);
  if (!std::isfinite(length) || samples.size() >= kMaxSamples)
    return false;
  if (length <= spacing) {
    samples.push_back(Translation(b.back()));
    return true;
  }
  if (depth == 24)
    return false;
  const auto [left, right] = Split(b);
  return Sample(left, spacing, samples, depth + 1) &&
         Sample(right, spacing, samples, depth + 1);
}

// Solve a clamped cubic spline in chord-length parameter. Refinement creates
// uneven knot spacing; treating every interval as equal can keep producing
// overshoot even as points are inserted. Scale Hermite derivatives back to each
// segment's local [0, 1] parameter after solving for shared dP/ds derivatives.
auto Fit(const std::vector<Knot>& knots)
    -> std::vector<frc::CubicHermiteSpline> {
  const size_t count = knots.size();
  std::vector<double> h;
  for (size_t i = 1; i < count; ++i)
    h.push_back(knots[i].point.Distance(knots[i - 1].point).value());
  std::vector<double> lower(count, 0), diagonal(count, 1), upper(count, 0);
  std::vector<cv::Point2d> rhs(count), derivative(count);
  rhs.front() = (Point(knots[1].point) - Point(knots[0].point)) * (1.2 / h[0]);
  rhs.back() = (Point(knots.back().point) - Point(knots[count - 2].point)) *
               (1.2 / h.back());
  for (size_t i = 1; i + 1 < count; ++i) {
    lower[i] = h[i];
    diagonal[i] = 2 * (h[i - 1] + h[i]);
    upper[i] = h[i - 1];
    rhs[i] = ((Point(knots[i].point) - Point(knots[i - 1].point)) *
                  (h[i] / h[i - 1]) +
              (Point(knots[i + 1].point) - Point(knots[i].point)) *
                  (h[i - 1] / h[i])) *
             3;
  }
  for (size_t i = 1; i < count; ++i) {
    const double factor = lower[i] / diagonal[i - 1];
    diagonal[i] -= factor * upper[i - 1];
    rhs[i] -= rhs[i - 1] * factor;
  }
  derivative.back() = rhs.back() / diagonal.back();
  for (size_t i = count - 1; i-- > 0;)
    derivative[i] = (rhs[i] - derivative[i + 1] * upper[i]) / diagonal[i];
  std::vector<frc::CubicHermiteSpline> splines;
  for (size_t i = 0; i + 1 < count; ++i) {
    const auto a = Point(knots[i].point), b = Point(knots[i + 1].point);
    const auto da = derivative[i] * h[i], db = derivative[i + 1] * h[i];
    splines.emplace_back(
        wpi::array<double, 2>{a.x, da.x}, wpi::array<double, 2>{b.x, db.x},
        wpi::array<double, 2>{a.y, da.y}, wpi::array<double, 2>{b.y, db.y});
  }
  return splines;
}

// Insert a previously omitted raw vertex, or bisect along the original route.
// Knots retain source arc positions even when the fitted curve takes shortcuts.
auto Refine(const Knot& a, const Knot& b,
            const std::vector<frc::Translation2d>& raw,
            const std::vector<double>& along) -> Knot {
  const double middle = (a.along + b.along) / 2;
  size_t best = raw.size();
  double distance = std::numeric_limits<double>::infinity();
  for (size_t i = 1; i + 1 < raw.size(); ++i) {
    if (along[i] > a.along && along[i] < b.along &&
        std::abs(along[i] - middle) < distance) {
      best = i;
      distance = std::abs(along[i] - middle);
    }
  }
  if (best != raw.size())
    return {.point = raw[best], .along = along[best]};
  auto upper = std::ranges::upper_bound(along, middle);
  const size_t i =
      std::min(static_cast<size_t>(upper - along.begin()), raw.size() - 1);
  const double fraction = (middle - along[i - 1]) / (along[i] - along[i - 1]);
  return {.point = raw[i - 1] + (raw[i] - raw[i - 1]) * fraction,
          .along = middle};
}
}  // namespace

DijkstraSplineNode::DijkstraSplineNode(std::string_view pose_channel,
                                       std::string_view goal_channel,
                                       std::string_view output_channel,
                                       std::string_view navgrid_channel,
                                       SplineSettings settings)
    : pose_channel_(pose_channel),
      goal_channel_(goal_channel),
      output_channel_(output_channel),
      navgrid_channel_(navgrid_channel),
      settings_(settings),
      dependencies_({{pose_channel_, typeid(PoseMessage)},
                     {goal_channel_, typeid(PoseMessage)},
                     {navgrid_channel_, typeid(streamer::NavgridMessage)}}),
      publications_({{output_channel_, typeid(SplinePathMessage)}}) {
  if (pose_channel_.empty() || goal_channel_.empty() ||
      output_channel_.empty() || pose_channel_ == goal_channel_ ||
      output_channel_ == pose_channel_ || output_channel_ == goal_channel_ ||
      navgrid_channel_.empty() || navgrid_channel_ == pose_channel_ ||
      navgrid_channel_ == goal_channel_ ||
      navgrid_channel_ == output_channel_) {
    throw std::invalid_argument(
        "Spline channels must be nonempty and distinct");
  }
  if (!std::isfinite(settings.robot_radius_meters) ||
      settings.robot_radius_meters < 0 ||
      !std::isfinite(settings.turn_penalty) || settings.turn_penalty < 0 ||
      settings.max_control_points < 2 ||
      !std::isfinite(settings.sample_spacing.value()) ||
      settings.sample_spacing.value() <= 0) {
    throw std::invalid_argument("Invalid spline fitting settings");
  }
}

void DijkstraSplineNode::EnableSplineDebug(std::string_view output_path) {
  std::scoped_lock lock(mutex_);
  if (output_path.empty() || output_path == pose_channel_ ||
      output_path == goal_channel_ || output_path == output_channel_ ||
      output_path == navgrid_channel_) {
    throw std::invalid_argument(
        "Spline debug channel must be nonempty and distinct");
  }
  if (!debug_path_.empty()) {
    if (debug_path_ == output_path)
      return;
    throw std::logic_error("Spline debug channel is already configured");
  }
  debug_path_ = output_path;
  publications_.emplace_back(debug_path_, typeid(streamer::PolylineMessage));
}

auto DijkstraSplineNode::Calculate(const PoseMessage* pose,
                                   const PoseMessage* goal,
                                   const DijkstraPlanner& planner) const
    -> SplinePathMessage {
  SplinePathMessage output;
  if (!pose || !goal)
    return output;
  const auto& start = pose->goal_position;
  const auto& end = goal->goal_position;
  const auto cleared = planner.BlockedStartCells(start);
  if (!planner.IsValidStartPose(start, cleared) ||
      !planner.IsValidPose(end, cleared))
    return output;
  const auto start_cell = planner.CellFor(start);
  const auto route = planner.FindPath(start_cell, planner.CellFor(end), cleared,
                                      start.Translation());
  output.status = SplineStatus::NoPath;
  if (route.empty())
    return output;
  const auto append = [&](const frc::Translation2d& point) {
    if (output.raw_route.empty() ||
        point.Distance(output.raw_route.back()).value() != 0)
      output.raw_route.push_back(point);
  };
  append(start.Translation());
  // Same-cell paths need no detour through the center.
  if (route.size() > 1) {
    const size_t first = planner.IsFree(start_cell, cleared) ? 0 : 1;
    for (size_t i = first; i < route.size(); ++i)
      append(planner.Center(route[i]));
  }
  append(end.Translation());
  output.status = SplineStatus::FitFailed;
  if (output.raw_route.size() == 1) {
    const auto p = Point(output.raw_route.front());
    if (planner.IsBoxFree(p.x, p.y, p.x, p.y, cleared)) {
      output.status = SplineStatus::Ready;
      output.samples = output.waypoints = output.raw_route;
    }
    return output;
  }
  std::vector<double> along{0};
  for (size_t i = 1; i < output.raw_route.size(); ++i)
    along.push_back(
        along.back() +
        output.raw_route[i].Distance(output.raw_route[i - 1]).value());
  std::vector<Knot> knots{{output.raw_route.front(), 0}};
  for (size_t i = 0; i + 1 < output.raw_route.size();) {
    size_t next = output.raw_route.size() - 1;
    const auto a = Point(output.raw_route[i]);
    while (next > i) {
      const auto b = Point(output.raw_route[next]);
      if (planner.IsVisible(a.x, a.y, b.x, b.y, cleared))
        break;
      --next;
    }
    if (next == i || knots.size() == settings_.max_control_points)
      return output;
    knots.push_back({output.raw_route[next], along[next]});
    i = next;
  }
  for (;;) {
    output.waypoints.clear();
    for (const auto& knot : knots)
      output.waypoints.push_back(knot.point);
    auto splines = Fit(knots);
    std::vector<bool> unsafe;
    bool clear = true;
    for (const auto& spline : splines) {
      const bool collides = !IsClear(planner, Controls(spline), cleared);
      unsafe.push_back(collides);
      clear &= !collides;
    }
    if (clear) {
      output.samples.push_back(output.waypoints.front());
      for (const auto& spline : splines) {
        if (!Sample(Controls(spline), settings_.sample_spacing.value(),
                    output.samples)) {
          output.samples.clear();
          return output;
        }
      }
      output.splines = std::move(splines);
      output.status = SplineStatus::Ready;
      return output;
    }
    if (output.refinement_passes >= settings_.max_refinement_passes)
      return output;
    const size_t additions = std::count(unsafe.begin(), unsafe.end(), true);
    if (additions > settings_.max_control_points - knots.size())
      return output;
    std::vector<Knot> refined{knots.front()};
    for (size_t i = 0; i < unsafe.size(); ++i) {
      if (unsafe[i]) {
        auto point = Refine(knots[i], knots[i + 1], output.raw_route, along);
        if (point.along <= knots[i].along ||
            point.along >= knots[i + 1].along ||
            point.point.Distance(knots[i].point).value() == 0 ||
            point.point.Distance(knots[i + 1].point).value() == 0)
          return output;
        refined.push_back(point);
      }
      refined.push_back(knots[i + 1]);
    }
    knots = std::move(refined);
    ++output.refinement_passes;
  }
}

auto DijkstraSplineNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    {
      std::scoped_lock lock(mutex_);
      if (!context->Exists(pose_channel_) || !context->Exists(goal_channel_) ||
          !context->Exists(navgrid_channel_) ||
          context->Exists(output_channel_))
        return;
      auto navgrid =
          context->GetSharedMessage<streamer::NavgridMessage>(navgrid_channel_);
      // Publishers supply a new immutable snapshot when the grid changes.
      if (navgrid != navgrid_) {
        planner_.reset();
        navgrid_ = std::move(navgrid);
        if (navgrid_) {
          try {
            planner_ = std::make_shared<DijkstraPlanner>(
                *navgrid_, settings_.turn_penalty,
                settings_.robot_radius_meters);
          } catch (const std::invalid_argument&) {
            // Publish InvalidInput so the follower can issue a stop command.
          }
        }
      }
      auto output = std::make_shared<SplinePathMessage>();
      if (planner_) {
        *output = Calculate(context->GetMessage<PoseMessage>(pose_channel_),
                            context->GetMessage<PoseMessage>(goal_channel_),
                            *planner_);
        output->planner = planner_;
      }
      if (!debug_path_.empty()) {
        auto debug = std::make_shared<streamer::PolylineMessage>();
        if (planner_) {
          debug->field_size_x = planner_->GetGrid().field_size_x;
          debug->field_size_y = planner_->GetGrid().field_size_y;
        }
        debug->g = 0;
        for (const auto& point : output->samples)
          debug->points.push_back(Point(point));
        if (output->status == SplineStatus::Ready) {
          for (const auto& point : output->waypoints)
            debug->markers.push_back(Point(point));
        }
        context->SetMessage(debug_path_, std::move(debug));
      }
      context->SetMessage(output_channel_, std::move(output));
    }
    for (const auto& callback : callbacks_)
      callback(context);
  };
}

auto DijkstraSplineNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto DijkstraSplineNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void DijkstraSplineNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}
}  // namespace pathing
