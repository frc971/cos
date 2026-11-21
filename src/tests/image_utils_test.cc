#include "gamepiece/image_utils.h"

#include <utility>

#include <gtest/gtest.h>

namespace gamepiece {
namespace {

TEST(ImageUtilsTest, TwoPixelDilationMatchesDistanceTransformAtSensorEdges) {
  cv::RNG rng(71);
  for (const auto size : {cv::Size{1, 1}, cv::Size{1, 40}, cv::Size{40, 1}, cv::Size{128, 96}})
    for (const float threshold : {0.f, .02f, .5f, .98f, 1.f}) {
      cv::Mat1f noise(size), distance; rng.fill(noise, cv::RNG::UNIFORM, 0, 1);
      cv::Mat1b mask = noise < threshold, nearby;
      DilateTwoPixelL2(mask, nearby);
      cv::distanceTransform(255-mask, distance, cv::DIST_L2, 3);
      EXPECT_EQ(cv::countNonZero(nearby != (distance <= 2)), 0);
    }
}

template <typename T>
class ImageUtilsTest : public testing::Test {
 protected:
  static auto TestImage() -> cv::Mat3b {
    cv::Mat3b image(2, 3, cv::Vec3b{0, 0, 0});
    image(0, 1) = cv::Vec3b{0, 255, 255};
    image(1, 2) = cv::Vec3b{0, 255, 255};
    image(1, 0) = cv::Vec3b{0, 0, 255};
    return image;
  }
};
using PointTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(ImageUtilsTest, PointTypes);

TYPED_TEST(ImageUtilsTest, CpuThresholdConvertsIntegerPixelLocations) {
  std::vector<cv::Point_<TypeParam>> points;
  cv::Mat3b hsv;
  cv::Mat1b mask;
  HSVThreshold(TestFixture::TestImage(), 20, 40, 150, points, hsv, mask);
  const std::vector<cv::Point_<TypeParam>> expected{{1, 0}, {2, 1}};
  EXPECT_EQ(points, expected);
  HSVThreshold(TestFixture::TestImage(), 60, 70, 150, points, hsv, mask);
  EXPECT_TRUE(points.empty());
}

TYPED_TEST(ImageUtilsTest, GpuThresholdMatchesCpuAndCompletesStream) {
  ASSERT_GT(cv::cuda::getCudaEnabledDeviceCount(), 0);
  cv::cuda::Stream stream;
  cv::cuda::GpuMat image;
  const cv::Mat3b cpu_image = TestFixture::TestImage();
  image.upload(cpu_image, stream);
  cv::cuda::GpuMat hsv;
  cv::cuda::GpuMat mask;
  std::vector<cv::Point_<TypeParam>> points;
  HSVThreshold(image, 20, 40, 150, points, hsv, mask, stream);
  const std::vector<cv::Point_<TypeParam>> expected{{1, 0}, {2, 1}};
  EXPECT_EQ(points, expected);
  EXPECT_TRUE(stream.queryIfComplete());

  HSVThreshold(image, 60, 70, 150, hsv, mask, stream);
  cv::Mat1b cpu_mask;
  mask.download(cpu_mask, stream);
  stream.waitForCompletion();
  EXPECT_EQ(cv::countNonZero(cpu_mask), 0);
  HSVThreshold(image, 60, 70, 150, points, hsv, mask, stream);
  EXPECT_TRUE(points.empty());
}

TYPED_TEST(ImageUtilsTest, PinholeNormalizesUsingIntrinsics) {
  cv::Matx44d extrinsics = cv::Matx44d::eye();
  extrinsics(1, 3) = -2;
  const cv::Matx33d intrinsics{100, 0, 320, 0, 200, 240, 0, 0, 1};
  const auto offset = DistortedPinholePointOffset(
      cv::Point_<TypeParam>{370, 440}, 0, extrinsics, intrinsics);
  EXPECT_DOUBLE_EQ(offset.X().value(), 2);
  EXPECT_DOUBLE_EQ(offset.Y().value(), -1);
  const auto float_matrix_offset = UndistortedPinholePointOffset(
      cv::Point_<TypeParam>{0.5, 1}, 0, cv::Matx44f(extrinsics));
  EXPECT_EQ(offset, float_matrix_offset);
}

TYPED_TEST(ImageUtilsTest, CullsHorizonBeforeProjection) {
  cv::Matx44d extrinsics = cv::Matx44d::eye();
  extrinsics(1, 0) = 0.5;  // Rolled horizon: y = -0.5*x.
  extrinsics(1, 3) = -2;
  std::vector<cv::Point_<TypeParam>> points{
      {0, -1}, {0, 0}, {2, -1}, {2, 0}, {-2, 2},
      {std::numeric_limits<TypeParam>::quiet_NaN(), 1}};
  CullAboveHorizon(points, extrinsics);
  const std::vector<cv::Point_<TypeParam>> expected{{2, 0}, {-2, 2}};
  EXPECT_EQ(points, expected);
  for (const auto& point : points)
    EXPECT_DOUBLE_EQ(UndistortedPinholePointOffset(point, 0, extrinsics)
                         .X().value(), 2);
  CullAboveHorizon(points, cv::Matx44f(extrinsics));
  EXPECT_EQ(points, expected);
  points.clear();
  CullAboveHorizon(points, extrinsics);
  EXPECT_TRUE(points.empty());
}

TEST(ImageUtilsTest, CullsPairedPixelsWithoutLosingAlignment) {
  using Pixel = std::pair<cv::Point2d, cv::Point2d>;
  std::vector<Pixel> pixels{{{0, -1}, {0, 1}}, {{1, 1}, {1, 0}},
                            {{2, 1}, {2, 2}}};
  CullAboveHorizon(pixels, cv::Matx44d::eye(), &Pixel::first);
  CullAboveHorizon(pixels, cv::Matx44d::eye(), &Pixel::second);
  const std::vector<Pixel> expected{{{2, 1}, {2, 2}}};
  EXPECT_EQ(pixels, expected);
}

}  // namespace
}  // namespace gamepiece
