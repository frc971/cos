#include "gamepiece/image_utils.h"

#include <gtest/gtest.h>

namespace gamepiece {
namespace {

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
  ASSERT_TRUE(offset.has_value());
  EXPECT_DOUBLE_EQ(offset->X().value(), 2);
  EXPECT_DOUBLE_EQ(offset->Y().value(), -1);
  const auto float_matrix_offset = UndistortedPinholePointOffset(
      cv::Point_<TypeParam>{0.5, 1}, 0, cv::Matx44f(extrinsics));
  ASSERT_TRUE(float_matrix_offset.has_value());
  EXPECT_EQ(*offset, *float_matrix_offset);
  EXPECT_FALSE(UndistortedPinholePointOffset(
      cv::Point_<TypeParam>{0, 0}, 0, extrinsics).has_value());
  EXPECT_FALSE(UndistortedPinholePointOffset(
      cv::Point_<TypeParam>{0, -1}, 0, extrinsics).has_value());
}

}  // namespace
}  // namespace gamepiece
