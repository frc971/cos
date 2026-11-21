#include <cstdlib>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include "gamepiece/unet_mask.h"

// Needs COS_UNET_ENGINE (TensorRT engine) and COS_UNET_IMAGE (BGR frame); skipped otherwise.
TEST(UnetMask, MonotonicBinaryFullResolutionMasks) {
  const char *engine = std::getenv("COS_UNET_ENGINE"), *path = std::getenv("COS_UNET_IMAGE");
  if (!engine || !path) GTEST_SKIP() << "Set COS_UNET_ENGINE and COS_UNET_IMAGE";
  const cv::Mat image = cv::imread(path);
  ASSERT_FALSE(image.empty());
  gamepiece::UnetMask unet(engine);
  cv::Mat1b strict, loose, extra;
  unet.Threshold(image, .5f, strict);
  unet.Threshold(image, .05f, loose);
  ASSERT_EQ(loose.size(), image.size());
  EXPECT_EQ(cv::countNonZero((strict != 0) & (strict != 255)), 0);
  cv::bitwise_and(strict, ~loose, extra);
  EXPECT_EQ(cv::countNonZero(extra), 0) << "A lower confidence must keep every strict pixel";
  EXPECT_GT(cv::countNonZero(loose), 0);
  EXPECT_GE(cv::countNonZero(loose), cv::countNonZero(strict));
}
