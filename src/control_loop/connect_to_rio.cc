#include "control_loop/connect_to_rio.h"

#include "utils/stop.h"

#include "absl/log/log.h"

#include <networktables/NetworkTableInstance.h>
#include <networktables/StringTopic.h>
#include <chrono>
#include <filesystem>
#include <thread>

namespace control_loop {
namespace {
// Publishes logname such as log32 to networktables so we can easily find match logs
void PublishLogName(const std::string& path) {
  static auto log_name_publisher = nt::NetworkTableInstance::GetDefault()
                                       .GetTable("COS")
                                       ->GetStringTopic("LogName")
                                       .Publish();
  log_name_publisher.Set(path);
}
}  // namespace

auto GetLogPath() -> const std::string& {
  static const std::string log_path = GetNewLogPath();
  return log_path;
}

void StartNetworktables(int team_number) {
  nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
  inst.StopServer();
  inst.StopClient();
  inst.StartClient4("orin_localization");
  inst.SetServerTeam(team_number);
  inst.StartDSClient();
  const std::string& log_path = GetLogPath();
  LOG(INFO) << "Log path: " << log_path;

  LOG(INFO) << "Team number: " << team_number;
  LOG(INFO) << "Waiting for connection and time synchronization";
  while ((!inst.IsConnected() || !inst.GetServerTimeOffset()) &&
         !stop::StopRequested()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (stop::StopRequested()) {
    LOG(INFO) << "Stopped while waiting for connection to rio";
    return;
  }

  PublishLogName(log_path);
  LOG(INFO) << "Connected to rio!";
}

void StartNetworktablesAsHost() {
  nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
  inst.StopLocal();
  inst.StopClient();
  inst.StartServer("orin_localization");
  const std::string& log_path = GetLogPath();
  LOG(INFO) << "Log path: " << log_path;
  PublishLogName(log_path);
}

auto GetNewLogPath(const std::string& log_dir) -> std::string {
  int id = 0;
  std::filesystem::path log_path;
  do {
    log_path = std::filesystem::path(log_dir) / ("log" + std::to_string(id));
    id++;
  } while (std::filesystem::exists(log_path));
  std::filesystem::create_directories(log_path);
  return log_path.string();
}
}  // namespace control_loop
