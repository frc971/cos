#include "gamepiece/image_utils.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace gamepiece {

void DilateTwoPixelL2(const cv::Mat1b& mask, cv::Mat1b& nearby) {
  static const cv::Mat1b kernel = (cv::Mat1b(5, 5) <<
      0, 0, 1, 0, 0,
      0, 1, 1, 1, 0,
      1, 1, 1, 1, 1,
      0, 1, 1, 1, 0,
      0, 0, 1, 0, 0);
  cv::dilate(mask, nearby, kernel);
}

void HSVThreshold(const cv::cuda::GpuMat& bgr_image, const int minimum_hue,
                  const int maximum_hue, const int minimum_saturation,
                  cv::cuda::GpuMat& hsv_image, cv::cuda::GpuMat& threshold_mask,
                  cv::cuda::Stream& stream) {
  cv::cuda::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV, 0, stream);
  cv::cuda::inRange(hsv_image, cv::Scalar(minimum_hue, minimum_saturation, 0),
                    cv::Scalar(maximum_hue, 255, 255), threshold_mask, stream);
}

}  // namespace gamepiece
