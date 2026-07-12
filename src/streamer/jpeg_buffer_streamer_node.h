#pragma once
#include <memory>
#include <nadjieb/mjpeg_streamer.hpp>
#include "camera/camera.h"

namespace streamer {

class JpegBufferStreamerNode {
 public:
  JpegBufferStreamerNode(std::string path, int port);
  void Stream(const std::shared_ptr<camera::JpegBuffer>& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string path_;
};

}  // namespace streamer
