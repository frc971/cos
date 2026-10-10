#pragma once
#include <string>
namespace control_loop {
// Sets up NetworkTables and publishes the run log directory.
// Call at the beginning of every robot main; WPILogWriter records publications.
void StartNetworktables(int team_number = 971);
void StartNetworktablesAsHost();
// Uses /AdvantageKit/DriverStation/Enabled and logs from the first enable
// through the second disable. Missing/disconnected data stops and resets it.
auto ShouldLogImages() -> bool;
// Creates the log directory on first use and returns the same path thereafter.
auto GetLogPath() -> const std::string&;
auto GetNewLogPath(const std::string& log_dir = "/cos/logs") -> std::string;
}  // namespace control_loop
