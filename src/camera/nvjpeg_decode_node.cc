#include "camera/nvjpeg_decode_node.h"
#include "control_loop/timer.h"

#include <array>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {
using std::function;

namespace {

auto CheckCuda(cudaError_t status) -> void {
  CHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

auto ConfigureDestination(DecodedJpegBuffer* buffer,
                          nvjpegOutputFormat_t output_format, int components,
                          const std::array<int, NVJPEG_MAX_COMPONENT>& widths,
                          const std::array<int, NVJPEG_MAX_COMPONENT>& heights)
    -> void {
  buffer->output_format = output_format;
  buffer->width = widths[0];
  buffer->height = heights[0];

  auto set_channel = [buffer](int channel, size_t pitch, int height) -> void {
    buffer->destination.pitch[channel] = pitch;
    buffer->channel_sizes[channel] = pitch * static_cast<size_t>(height);
    buffer->output_size += buffer->channel_sizes[channel];
  };

  switch (output_format) {
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI:
      set_channel(0, static_cast<size_t>(widths[0]) * 3U, heights[0]);
      break;
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
      for (int channel = 0; channel < 3; ++channel) {
        set_channel(channel, static_cast<size_t>(widths[0]), heights[0]);
      }
      break;
    case NVJPEG_OUTPUT_Y:
      set_channel(0, static_cast<size_t>(widths[0]), heights[0]);
      break;
    case NVJPEG_OUTPUT_YUV:
    case NVJPEG_OUTPUT_UNCHANGED:
      for (int channel = 0; channel < components; ++channel) {
        set_channel(channel, static_cast<size_t>(widths[channel]),
                    heights[channel]);
      }
      break;
    case NVJPEG_OUTPUT_NV12:
      set_channel(0, static_cast<size_t>(widths[0]), heights[0]);
      set_channel(1, static_cast<size_t>(widths[1]) * 2U, heights[1]);
      break;
    case NVJPEG_OUTPUT_YUY2:
      set_channel(0, static_cast<size_t>(widths[0]) * 2U, heights[0]);
      break;
    case NVJPEG_OUTPUT_UNCHANGEDI:
      set_channel(
          0, static_cast<size_t>(widths[0]) * static_cast<size_t>(components),
          heights[0]);
      break;
    case NVJPEG_OUTPUT_UNCHANGEDI_U16:
      set_channel(0,
                  static_cast<size_t>(widths[0]) *
                      static_cast<size_t>(components) * sizeof(unsigned short),
                  heights[0]);
      break;
    default:
      LOG(FATAL) << "Unsupported nvJPEG output format: " << output_format;
  }

  buffer->stride = buffer->destination.pitch[0];
}

}  // namespace

DecodedJpegBuffer::~DecodedJpegBuffer() {
  for (int channel = 0; channel < NVJPEG_MAX_COMPONENT; ++channel) {
    if (channel_sizes[channel] != 0U &&
        destination.channel[channel] != nullptr) {
      cudaError_t status = cudaFree(destination.channel[channel]);
      if (status != cudaSuccess) {
        LOG(ERROR) << cudaGetErrorString(status);
      }
    }
  }
}

DecodedJpegBuffer::DecodedJpegBuffer(DecodedJpegBuffer&& other) noexcept
    : width(other.width),
      height(other.height),
      stride(other.stride),
      output_size(other.output_size),
      timestamp(other.timestamp),
      output_format(other.output_format),
      channel_sizes(other.channel_sizes),
      destination(other.destination) {
  other.channel_sizes = {};
  other.destination = {};
  other.output_size = 0;
}

NvjpegDecodeNode::NvjpegDecodeNode(std::string_view input_path,
                                   std::string_view output_path,
                                   nvjpegOutputFormat_t output_format,
                                   control_loop::ThreadPool& thread_pool)
    : input_path_({std::string(input_path)}),
      output_path_(output_path),
      output_format_(output_format),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(JpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          DecodedJpegBuffer>(output_path_)}) {
  CheckCuda(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  CHECK(nvjpegCreateSimple(&handle_) == NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecoderStateCreate(handle_, decoder_, &state_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegJpegStreamCreate(handle_, &jpeg_stream_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeParamsCreate(handle_, &decode_params_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegBufferPinnedCreate(handle_, nullptr, &pinned_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegBufferDeviceCreate(handle_, nullptr, &device_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegStateAttachPinnedBuffer(state_, pinned_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegStateAttachDeviceBuffer(state_, device_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  cudaStreamCreate(&stream_);
}

NvjpegDecodeNode::~NvjpegDecodeNode() {
  LOG(INFO) << "Destructing NvjpegDecodeNode";
  if (stream_ != nullptr) {
    CheckCuda(cudaStreamDestroy(stream_));
  }
  if (device_buffer_ != nullptr) {
    CHECK(nvjpegBufferDeviceDestroy(device_buffer_) == NVJPEG_STATUS_SUCCESS);
  }
  if (pinned_buffer_ != nullptr) {
    CHECK(nvjpegBufferPinnedDestroy(pinned_buffer_) == NVJPEG_STATUS_SUCCESS);
  }
  if (decode_params_ != nullptr) {
    CHECK(nvjpegDecodeParamsDestroy(decode_params_) == NVJPEG_STATUS_SUCCESS);
  }
  if (jpeg_stream_ != nullptr) {
    CHECK(nvjpegJpegStreamDestroy(jpeg_stream_) == NVJPEG_STATUS_SUCCESS);
  }
  if (state_ != nullptr) {
    CHECK(nvjpegJpegStateDestroy(state_) == NVJPEG_STATUS_SUCCESS);
  }
  if (decoder_ != nullptr) {
    CHECK(nvjpegDecoderDestroy(decoder_) == NVJPEG_STATUS_SUCCESS);
  }
  if (handle_ != nullptr) {
    CHECK(nvjpegDestroy(handle_) == NVJPEG_STATUS_SUCCESS);
  }
}

auto NvjpegDecodeNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    bool exists;
    auto* jpeg_buffer = context->GetMessage<JpegBuffer>(input_path_, exists);
    CHECK(exists) << input_path_;
    if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr ||
        jpeg_buffer->size == 0U) {
      context->SetMessage(output_path_, nullptr);
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }

    std::function<void()> task = [this, context, jpeg_buffer]() -> void {
      control_loop::Timer timer;
      auto decoded = DecodeJpegBuffer(jpeg_buffer);
      std::unique_ptr<control_loop::IMessage> decoded_buffer;
      if (decoded.has_value()) {
        decoded_buffer =
            std::make_unique<DecodedJpegBuffer>(std::move(*decoded));
      }

      context->SetMessage(output_path_, std::move(decoded_buffer));
      if (latency_channel_.has_value()) {
        context->SetMessage(
            latency_channel_.value(),
            std::make_unique<control_loop::LatencyMessage>(timer.Stop()));
      }

      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    thread_pool_.Submit(task, context->id);
  };
}

auto NvjpegDecodeNode::DecodeJpegBuffer(const JpegBuffer* const jpeg_buffer)
    -> std::optional<DecodedJpegBuffer> {
  std::scoped_lock lock(decode_mutex_);

  auto succeeded = [this, jpeg_buffer](nvjpegStatus_t status,
                                      const char* operation) -> bool {
    if (status == NVJPEG_STATUS_SUCCESS) {
      return true;
    }
    LOG(WARNING) << "Dropping undecodable JPEG on " << input_path_
                 << ": size=" << jpeg_buffer->size
                 << " timestamp=" << jpeg_buffer->timestamp
                 << ": " << operation << " failed with nvJPEG status " << status;
    return false;
  };

  int components = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  std::array<int, NVJPEG_MAX_COMPONENT> widths = {};
  std::array<int, NVJPEG_MAX_COMPONENT> heights = {};
  const auto* jpeg_data = static_cast<unsigned char*>(jpeg_buffer->ptr);

  if (!succeeded(
          nvjpegGetImageInfo(handle_, jpeg_data, jpeg_buffer->size, &components,
                             &subsampling, widths.data(), heights.data()),
          "nvjpegGetImageInfo")) {
    return std::nullopt;
  }

  if (!succeeded(nvjpegJpegStreamParse(handle_, jpeg_data, jpeg_buffer->size,
                                      0, 0, jpeg_stream_),
                 "nvjpegJpegStreamParse") ||
      !succeeded(nvjpegDecodeJpegHost(handle_, decoder_, state_, decode_params_,
                                     jpeg_stream_),
                 "nvjpegDecodeJpegHost")) {
    return std::nullopt;
  }

  DecodedJpegBuffer decoded_buffer{};
  decoded_buffer.timestamp = jpeg_buffer->timestamp;

  ConfigureDestination(&decoded_buffer, output_format_, components, widths,
                       heights);

  for (int channel = 0; channel < NVJPEG_MAX_COMPONENT; ++channel) {
    if (decoded_buffer.channel_sizes[channel] == 0U) {
      continue;
    }
    CheckCuda(cudaMalloc(
        reinterpret_cast<void**>(&decoded_buffer.destination.channel[channel]),
        decoded_buffer.channel_sizes[channel]));
  }

  if (!succeeded(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_,
                                                 jpeg_stream_, nullptr),
                 "nvjpegDecodeJpegTransferToDevice")) {
    return std::nullopt;
  }
  const auto decode_status = nvjpegDecodeJpegDevice(
      handle_, decoder_, state_, &decoded_buffer.destination, stream_);
  CheckCuda(cudaStreamSynchronize(stream_));
  if (!succeeded(decode_status, "nvjpegDecodeJpegDevice")) {
    return std::nullopt;
  }

  return decoded_buffer;
}

auto NvjpegDecodeNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto NvjpegDecodeNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void NvjpegDecodeNode::EnableTiming(std::string_view latency_channel) {
  publications_.push_back(control_loop::MessageDescriptor::Publication<
                          control_loop::LatencyMessage>(latency_channel));
  latency_channel_ = latency_channel;
}

}  // namespace camera
