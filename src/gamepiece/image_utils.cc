#include "gamepiece/image_utils.h"

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

}  // namespace gamepiece
