#include "absl/log/check.h"
#include "absl/log/log.h"

#include "camera/uvc_camera_node.h"

namespace camera {

UVCCameraNode::UVCCameraNode(const UVCCameraConfig& config)
    : name_(config.name) {
  {
    int code = uvc_init(&context_, nullptr);
    CHECK(!code) << "UVC failed to init will error code: " << code;
  }
  {
    int code =
        uvc_find_device(context_, &device_, 0, 0, config.serial_id.c_str());
    CHECK(!code) << "UVC failed to find device with error code: " << code
                 << " camera_name: " << config.name;
  }
  {
    int code = uvc_open(device_, &device_handle_);
    CHECK(!code) << "UVC failed to open device with error code: " << code
                 << " camera name: " << config.name;
  }
  {
    int code = uvc_get_stream_ctrl_format_size(
        device_handle_, &ctrl_, UVC_FRAME_FORMAT_MJPEG, config.width,
        config.height, config.fps);
    CHECK(!code) << "UVC failed to get stream ctrl format with exit code: "
                 << code << " camera_name: " << config.name;

    ctrl_.dwMaxPayloadTransferSize = config.max_payload_size;
    ctrl_.dwMaxVideoFrameSize = config.max_frame_size;
  }
}

void UVCCameraNode::CallBack(uvc_frame_t* frame) {
  CHECK(frame->frame_format == UVC_COLOR_FORMAT_MJPEG);
  std::shared_ptr<JpegBuffer> buffer =
      std::make_shared<JpegBuffer>(frame->data_bytes);
  std::memcpy(buffer->ptr(), frame->data, frame->data_bytes);

  for (int i = 0; i < callbacks_.size(); i++) {  // NOLINT
    callbacks_[i](buffer);
  }
}

void UVCCameraNode::Start() {
  int code = uvc_start_streaming(
      device_handle_, &ctrl_,
      [](uvc_frame_t* frame, void* ptr) {
        auto uvc_camera_node = static_cast<UVCCameraNode*>(ptr);
        uvc_camera_node->CallBack(frame);
      },
      this, 0);
  CHECK(!code) << "UVC failed to start streaming with exit code: " << code
               << " camera name: " << name_;
}

UVCCameraNode::~UVCCameraNode() {
  uvc_stop_streaming(device_handle_);
  uvc_close(device_handle_);
  uvc_unref_device(device_);
  uvc_exit(context_);
  LOG(INFO) << name_ << " has been destructed";
}

void UVCCameraNode::RegisterCallback(
    const std::function<void(std::shared_ptr<JpegBuffer>)>& callback) {
  callbacks_.push_back(callback);
}
}  // namespace camera
