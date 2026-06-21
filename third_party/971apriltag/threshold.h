#ifndef FRC_ORIN_THRESHOLD_H_
#define FRC_ORIN_THRESHOLD_H_

#include <stdint.h>

#include "apriltag_types.h"
#include "cuda.h"
// #include "vision_generated.h"

// TODO Charlie
namespace vision {
enum ImageFormat {
  MONO8 = 0,
  MONO16 = 1,
  YUYV422 = 2,
  BGR8 = 3,
  BGRA8 = 4,
  MJPEG = 5,
};
}

namespace frc::apriltag {

// Returns the number of bytes per pixel.
constexpr __device__ apriltag_size_t
BytesPerPixel(const vision::ImageFormat image_format) {
  switch (image_format) {
    case vision::ImageFormat::MONO8:
      return 1;
    case vision::ImageFormat::MONO16:
      return 2;
    case vision::ImageFormat::YUYV422:
      return 2;
    case vision::ImageFormat::BGR8:
      return 3;
    case vision::ImageFormat::BGRA8:
      return 4;
    case vision::ImageFormat::MJPEG:
      return 0;  // MJPEG not supported
  }
  return 0;
}

class Threshold {
 public:
  // Create a full-size grayscale image from a color image on the provided
  // stream.
  virtual void ToGreyscale(const uint8_t* color_image, uint8_t* gray_image,
                           CudaStream* stream) = 0;

  // Converts to grayscale, decimates, and thresholds an image on the provided
  // stream.
  virtual void ThresholdAndDecimate(const uint8_t* color_image,
                                    uint8_t* decimated_image,
                                    uint8_t* thresholded_image,
                                    apriltag_size_t min_white_black_diff,
                                    CudaStream* stream) = 0;

  virtual ~Threshold() = default;
};

// Constructs an optimized threshold object for the provided image format.
std::unique_ptr<Threshold> MakeThreshold(vision::ImageFormat image_format,
                                         size_t width, size_t height);

// Constructs a NEON optimized threshold object for the provided image format.
std::unique_ptr<Threshold> MakeNeonThreshold(vision::ImageFormat image_format,
                                             size_t width, size_t height);

}  // namespace frc::apriltag

#endif  // FRC_ORIN_THRESHOLD_H_
