#include <fstream>
#include <utility>

#include "camera/jpeg_buffer.h"
#include "logging/jpeg_buffer_log_node.h"

namespace logging {

JpegBufferLogNode::JpegBufferLogNode(std::string_view input_channel,
                                     std::string_view folder_path,
                                     control_loop::ThreadPool& thread_pool,
                                     std::function<bool()> should_log)
    : input_channel_(input_channel),
      folder_path_(folder_path),
      dependencies_({{input_channel_, typeid(camera::JpegBuffer)}}),
      thread_pool_(thread_pool),
      should_log_(std::move(should_log)) {}

auto JpegBufferLogNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    thread_pool_.Submit(
        [this, context]() -> void {
          auto jpeg_buffer =
              context->GetMessage<camera::JpegBuffer>(input_channel_);
          std::string output_file_path =
              folder_path_ + "/" + std::to_string(jpeg_buffer->timestamp) +
              ".jpg";
          std::ofstream out(output_file_path, std::ios::binary);
          out.write(reinterpret_cast<const char*>(jpeg_buffer->ptr),
                    jpeg_buffer->size);
          for (const auto& callback : callbacks_) {
            callback(context);
          }
        },
        context->id);
  };
}

auto JpegBufferLogNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto JpegBufferLogNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void JpegBufferLogNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

}  // namespace logging
