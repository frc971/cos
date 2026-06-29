#include <sstream>
#include <type_traits>

#include "gtest/gtest.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position_solver.h"
#include "localization/square_solver_node.h"
#include "localization/unambiguous_solver_node.h"

namespace {

TEST(PositionEstimateTest, StreamsPoseAndSummaryFields) {
  localization::position_estimate_t estimate{
      .pose = wpi::math::Pose3d{wpi::units::meter_t{1.0},
                                wpi::units::meter_t{2.0},
                                wpi::units::meter_t{3.0},
                                wpi::math::Rotation3d{}},
      .variance = 0.25,
      .num_tags = 2};

  std::ostringstream out;
  out << estimate;

  EXPECT_NE(out.str().find("pose(x=1"), std::string::npos);
  EXPECT_NE(out.str().find("variance=0.25"), std::string::npos);
  EXPECT_NE(out.str().find("num_tags=2"), std::string::npos);
}

TEST(PositionSolverTest, VarianceScalesByDistanceAndTagCount) {
  EXPECT_DOUBLE_EQ(localization::Variance(2, 8.0, 1.0, 0.5), 2.0);
}

TEST(PositionSolverInterfacesTest, ConcreteSolversImplementInterfaces) {
  EXPECT_TRUE((std::is_base_of_v<localization::IPositionSolverNode,
                                 localization::SquareSolverNode>));
  EXPECT_TRUE((std::is_base_of_v<localization::IPositionSolverNode,
                                 localization::MultiTagSolverNode>));
  EXPECT_TRUE((std::is_base_of_v<localization::IAccumulatingSolverNode,
                                 localization::UnambiguousSolverNode>));
}

}  // namespace
