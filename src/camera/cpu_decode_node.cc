#include "camera/cpu_decode_node.h"

#include <jpeglib.h>

#include <cstddef>
#include <memory>

#include "absl/log/check.h"

namespace camera {

CpuJpegDecodeNode::CpuJpegDecodeNode(std::string_view input_path,
                                     std::string_view output_path,
                                     control_loop::ThreadPool& thread_pool)
    : input_path_(input_path),
      output_path_(output_path),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(JpegBuffer)}}),
      publications_({control_loop::MessageDescriptor::Publication<
          DecodedImageBuffer>(output_path_)}) {}

auto CpuJpegDecodeNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto notify_callbacks = [this, &context] -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    CHECK(context->Exists(input_path_));
    const auto* jpeg = context->GetMessage<JpegBuffer>(input_path_);
    if (jpeg == nullptr || jpeg->ptr == nullptr || jpeg->size == 0U) {
      context->SetMessage(output_path_, nullptr);
      notify_callbacks();
      return;
    }

    thread_pool_.Submit(
        [this, context, jpeg] -> void {
          auto decoded = std::make_unique<DecodedImageBuffer>(Decode(jpeg));
          context->SetMessage(output_path_, std::move(decoded));
          for (const auto& callback : callbacks_) {
            callback(context);
          }
        },
        context->id);
  };
}

auto CpuJpegDecodeNode::Decode(const JpegBuffer* jpeg_buffer)
    -> DecodedImageBuffer {
  DecodedImageBuffer decoded;
  jpeg_decompress_struct cinfo{};
  jpeg_error_mgr error{};
  cinfo.err = jpeg_std_error(&error);
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, static_cast<unsigned char*>(jpeg_buffer->ptr),
               jpeg_buffer->size);
  CHECK_EQ(jpeg_read_header(&cinfo, TRUE), JPEG_HEADER_OK);
  cinfo.out_color_space = JCS_GRAYSCALE;
  CHECK(jpeg_start_decompress(&cinfo));
  decoded.width = static_cast<int>(cinfo.output_width);
  decoded.height = static_cast<int>(cinfo.output_height);
  CHECK_EQ(cinfo.output_components, 1);
  decoded.stride = static_cast<size_t>(decoded.width);
  decoded.data.resize(decoded.stride * static_cast<size_t>(decoded.height));
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPLE* row = decoded.data.data() +
                   static_cast<size_t>(cinfo.output_scanline) * decoded.stride;
    CHECK_EQ(jpeg_read_scanlines(&cinfo, &row, 1), 1U);
  }
  CHECK(jpeg_finish_decompress(&cinfo));
  jpeg_destroy_decompress(&cinfo);
  return decoded;
}

auto CpuJpegDecodeNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto CpuJpegDecodeNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace camera
