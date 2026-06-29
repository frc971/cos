#include <filesystem>

#include "gtest/gtest.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/square_solver_node.h"
#include "localization/unambiguous_solver_node.h"
#include "unit_tests/test_helpers.h"

namespace {

class SolverFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto dir = std::filesystem::temp_directory_path();
    intrinsics_path = cos_test::testing::WriteJsonFile(
        dir / "solver_intrinsics_test.json", cos_test::testing::IntrinsicsJson());
    extrinsics_path = cos_test::testing::WriteJsonFile(
        dir / "solver_extrinsics_test.json", cos_test::testing::ExtrinsicsJson());
  }

  std::string intrinsics_path;
  std::string extrinsics_path;
};

TEST_F(SolverFixture, SquareSolverEmptyDetectionsProducesNoEstimates) {
  localization::SquareSolverNode solver(intrinsics_path, extrinsics_path);

  EXPECT_TRUE(solver.AmbiguousSolveWithoutNotify({}).empty());
}

TEST_F(SolverFixture, MultiTagSolverEmptyDetectionsProducesNoEstimate) {
  localization::MultiTagSolverNode solver(intrinsics_path, extrinsics_path);

  EXPECT_FALSE(solver.AmbiguousSolveWithoutNotify({}).has_value());
}

TEST_F(SolverFixture, UnambiguousSolverEmptyDetectionsProducesNoEstimate) {
  std::vector<camera::camera_constant_t> cameras = {
      cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                       extrinsics_path)};
  localization::UnambiguousSolverNode solver(cameras);

  EXPECT_FALSE(solver.SolveWithoutNotify({{}}).has_value());
}

TEST_F(SolverFixture, UnambiguousSolverAccumulateWaitsForAllCameras) {
  std::vector<camera::camera_constant_t> cameras = {
      cos_test::testing::MakeCameraConstant("front", intrinsics_path,
                                       extrinsics_path),
      cos_test::testing::MakeCameraConstant("rear", intrinsics_path,
                                       extrinsics_path)};
  localization::UnambiguousSolverNode solver(cameras);
  int callback_count = 0;
  solver.RegisterCallback(
      [&](localization::position_estimate_t,
          control_loops::MetaDataList,
          std::shared_ptr<control_loops::Context>) { callback_count++; });

  solver.Accumulate(std::make_shared<std::vector<apriltag::tag_detection_t>>(),
                    {{.camera_idx = 0, .timestamp = 1}}, nullptr);

  EXPECT_EQ(callback_count, 0);
}

}  // namespace
