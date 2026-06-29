#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

#include "camera/nvjpeg_decode_node.h"
#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"

namespace cos_test::testing {

inline auto RealApriltagFramePath() -> std::filesystem::path {
  return std::filesystem::path(COS_SOURCE_DIR) / "unit_tests" / "testdata" /
         "jpeg_frames" / "decoded" / "20.283378.png";
}

inline auto LoadRealApriltagFrame() -> cv::Mat {
  const std::filesystem::path frame_path = RealApriltagFramePath();
  cv::Mat gray = cv::imread(frame_path.string(), cv::IMREAD_GRAYSCALE);
  EXPECT_FALSE(gray.empty()) << "Missing decoded AprilTag fixture: "
                             << frame_path;
  EXPECT_TRUE(gray.empty() || gray.isContinuous());
  return gray;
}

inline auto MakeGrayNvBufferFrame(const cv::Mat& gray)
    -> std::shared_ptr<camera::DecodedJpegNvBuffer> {
  auto* nv_buffer =
      new NvBuffer(V4L2_PIX_FMT_GREY, static_cast<uint32_t>(gray.cols),
                   static_cast<uint32_t>(gray.rows), 0);
  nv_buffer->planes[0].data = gray.data;
  nv_buffer->planes[0].bytesused =
      static_cast<uint32_t>(gray.total() * gray.elemSize());
  nv_buffer->planes[0].length = nv_buffer->planes[0].bytesused;
  nv_buffer->planes[0].fmt.width = static_cast<uint32_t>(gray.cols);
  nv_buffer->planes[0].fmt.height = static_cast<uint32_t>(gray.rows);
  nv_buffer->planes[0].fmt.bytesperpixel = 1;
  nv_buffer->planes[0].fmt.stride = static_cast<uint32_t>(gray.step);
  nv_buffer->planes[0].fmt.sizeimage = nv_buffer->planes[0].bytesused;
  return std::make_shared<camera::DecodedJpegNvBuffer>(nv_buffer);
}

}  // namespace cos_test::testing
