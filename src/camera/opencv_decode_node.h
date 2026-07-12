#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "camera/decoded_jpeg_buffer.h"
#include "utils/node.h"

namespace camera {

// CPU-only JPEG decoder, backed by cv::imdecode instead of NVIDIA's hardware
// decoder. Produces the same DecodedJpegNvBuffer type as NvjpegDecodeNode
// (grayscale data in plane 0), so it is a drop-in replacement for it anywhere
// an IDecodeNode is expected -- in particular this lets the rest of the
// pipeline (apriltag detection, gamepiece detection) run entirely on CPU.
class OpenCVDecodeNode : public IDecodeNode {
 public:
  explicit OpenCVDecodeNode(const std::string& name);
  ~OpenCVDecodeNode() override;

  void RegisterCallback(
      const std::function<void(std::shared_ptr<DecodedJpegNvBuffer>,
                               control_loops::MetaDataList metadata,
                               std::shared_ptr<control_loops::Context>)>&
          callback) override;
  void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer,
              control_loops::MetaDataList metadata,
              std::shared_ptr<control_loops::Context> ctx) override;

 private:
  void DecodeJpegBuffer(const std::shared_ptr<JpegBuffer>& jpeg_buffer,
                        control_loops::MetaDataList metadata,
                        std::shared_ptr<control_loops::Context> ctx);

 private:
  std::string name_;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(std::shared_ptr<DecodedJpegNvBuffer>,
                                 control_loops::MetaDataList metadata,
                                 std::shared_ptr<control_loops::Context>)>>
      callbacks_;
  std::jthread decode_thread_;
};

}  // namespace camera
