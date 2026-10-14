#include "logging/wpilog_writer.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

#include <wpi/raw_ostream.h>

#include "camera/jpeg_buffer.h"
#include "localization/position.h"

namespace logging {

WPILogWriter::WPILogWriter(
    std::string_view filename,
    const std::vector<control_loop::MessageDescriptor>& publications,
    const nt::NetworkTableInstance& instance) {
  std::error_code error;
  log_ = std::make_unique<wpi::log::DataLogWriter>(filename, error);
  if (error) {
    throw std::system_error(error, "Cannot open WPILog file");
  }

  std::unordered_set<std::string> channels;
  std::unordered_set<std::string> seen_paths;
  std::vector<std::string> paths;
  // The file has one DataLogWriter, but every publication gets its own field
  // registrations. The channel is a runtime value from MessageDescriptor.
  for (const auto& publication : publications) {
    const auto& registration = publication.GetRegistration();
    if (!registration.has_value() || publication.GetTypes().size() != 1) {
      throw std::invalid_argument("Publication lacks concrete type: " +
                                  publication.GetChannel());
    }
    if (!channels.insert(publication.GetChannel()).second) {
      throw std::invalid_argument("Duplicate publication: " +
                                  publication.GetChannel());
    }

    if (*registration == nullptr) {
      throw std::invalid_argument("Missing WPILog registration: " +
                                  publication.GetChannel());
    }
    PublicationLog group{.channel = publication.GetChannel(), .append = {}};
    paths.clear();
    group.append = (*registration)(*log_, instance, group.channel, paths);
    for (const auto& path : paths) {
      if (!seen_paths.insert(path).second) {
        throw std::invalid_argument("Duplicate WPILog path: " + path);
      }
    }
    publications_.push_back(std::move(group));
  }

  flush_thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    std::unique_lock wait_lock(flush_wait_mutex_);
    while (!stop_token.stop_requested()) {
      flush_cv_.wait_for(wait_lock, stop_token, std::chrono::seconds(1),
                         []() -> bool { return false; });
      if (!stop_token.stop_requested()) Flush();
    }
  });
}

void WPILogWriter::Log(const control_loop::ContextInternal& context) {
  std::scoped_lock lock(mutex_);
  double total_capture_time = 0;
  std::size_t frame_count = 0;
  for (const auto* frame : context.GetMessages<camera::JpegBuffer>()) {
    if (std::isfinite(frame->timestamp) && frame->timestamp >= 0) {
      total_capture_time += frame->timestamp;
      ++frame_count;
    }
  }
  std::optional<double> capture_time;
  if (frame_count != 0) capture_time = total_capture_time / frame_count;
  for (auto& group : publications_) {
    const auto* message =
        context.GetMessage<control_loop::IMessage>(group.channel);
    if (message == nullptr) continue;
    auto sample_capture_time = capture_time;
    if (const auto* frame = dynamic_cast<const camera::JpegBuffer*>(message);
        frame != nullptr && std::isfinite(frame->timestamp) &&
        frame->timestamp >= 0) {
      sample_capture_time = frame->timestamp;
    } else if (const auto* pose =
                   dynamic_cast<const localization::PositionEstimateMessage*>(
                       message);
               pose != nullptr && std::isfinite(pose->timestamp) &&
               pose->timestamp > 0) {
      sample_capture_time = pose->timestamp;
    }
    if (!group.append(*message, sample_capture_time)) {
      throw std::runtime_error("WPILog message type mismatch: " +
                               group.channel);
    }
  }
}

void WPILogWriter::Flush() {
  std::scoped_lock lock(mutex_);
  log_->Flush();
  log_->GetStream().flush();
}

}  // namespace logging
