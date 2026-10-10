#pragma once

#include "control_loop/connect_to_rio.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"

namespace logging {

class JpegBufferLogNode final : public control_loop::INode {
 public:
  JpegBufferLogNode(
      std::string_view input_channel, std::string_view folder_path,
      control_loop::ThreadPool& thread_pool,
      std::function<bool()> should_log = control_loop::ShouldLogImages);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::string input_channel_;
  std::string folder_path_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  control_loop::ThreadPool& thread_pool_;
  std::function<bool()> should_log_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace logging
