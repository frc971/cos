#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "control_loop/control_loop.h"
#include "streamer/navgrid_streamer_node.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, navgrid,                    // NOLINT
          "/root/constants/assets/navgrid.json",   // NOLINT
          "Navgrid JSON file to stream.");         // NOLINT
ABSL_FLAG(std::string, field_image,                // NOLINT
          "/root/constants/assets/field.png",      // NOLINT
          "Field image to draw the navgrid on.");  // NOLINT
ABSL_FLAG(int, port, 4971, "MJPEG server port.");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  try {
    auto navgrid = std::make_shared<streamer::NavgridMessage>(
        absl::GetFlag(FLAGS_navgrid));
    streamer::NavgridStreamerNode streamer("navgrid", "/navgrid",
                                           absl::GetFlag(FLAGS_port),
                                           absl::GetFlag(FLAGS_field_image));
    auto stream = streamer.CreateCallback();
    control_loop::ControlLoop loop(std::chrono::milliseconds(100));
    loop.RegisterCallback([&](const control_loop::Context& context) -> void {
      context->SetMessage("navgrid", navgrid);
      stream(context);
    });
    loop.Start();
    stop::WaitUntilStop();
    loop.Stop();
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return 1;
  }
  return 0;
}
