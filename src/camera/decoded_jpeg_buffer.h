#pragma once
#include <functional>
#include <memory>

#include "NvBuffer.h"

#include "camera/camera.h"
#include "control_loops/context.h"
#include "utils/node.h"

namespace camera {

// A decoded JPEG frame. Backed by an NvBuffer so that decoders can be
// swapped between hardware (NvjpegDecodeNode) and software (OpenCVDecodeNode)
// implementations without changing anything downstream. NvBuffer itself has
// no CUDA/hardware dependency: it is a plain host-memory container, so
// software decoders can populate it directly with `new`-allocated planes.
class DecodedJpegNvBuffer {
 public:
  explicit DecodedJpegNvBuffer(NvBuffer* nv_buffer) : buffer(nv_buffer) {}
  ~DecodedJpegNvBuffer() { delete buffer; }
  NvBuffer* buffer;
};

// Interface implemented by every JPEG decode node (hardware or software) so
// that they can be used interchangeably anywhere a decoder is wired into a
// pipeline.
class IDecodeNode : public INode<std::shared_ptr<DecodedJpegNvBuffer>> {
 public:
  void RegisterCallback(
      const std::function<void(std::shared_ptr<DecodedJpegNvBuffer>,
                               control_loops::MetaDataList metadata,
                               std::shared_ptr<control_loops::Context>)>&
          callback) override = 0;
  virtual void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer,
                      control_loops::MetaDataList metadata,
                      std::shared_ptr<control_loops::Context> ctx) = 0;
  ~IDecodeNode() override = default;
};

}  // namespace camera
