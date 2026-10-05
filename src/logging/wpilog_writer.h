#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "control_loop/context.h"
#include "logging/log_registration.h"  // IWYU pragma: export

namespace logging {

class WPILogWriter {
 public:
  // Each field is appended to the file and published to the same NT path.
  WPILogWriter(std::string_view filename,
               const std::vector<control_loop::MessageDescriptor>& publications,
               const nt::NetworkTableInstance& instance =
                   nt::NetworkTableInstance::GetDefault());
  ~WPILogWriter() = default;

  WPILogWriter(const WPILogWriter&) = delete;
  auto operator=(const WPILogWriter&) -> WPILogWriter& = delete;

  void Log(const control_loop::ContextInternal& context);
  void Flush();

 private:
  struct PublicationLog {
    std::string channel;
    LogFunction append;
  };

  std::unique_ptr<wpi::log::DataLogWriter> log_;
  std::vector<PublicationLog> publications_;
  std::mutex mutex_;
  std::mutex flush_wait_mutex_;
  std::condition_variable_any flush_cv_;
  // Destroy the thread before the entries and their underlying log.
  std::jthread flush_thread_;
};

}  // namespace logging
