#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tools/jpeg_frame_tools.h"

ABSL_FLAG(std::string, log_path, "", "Byte log containing JPEG frames");
ABSL_FLAG(std::string, encoded_folder,
          (tools::DefaultFrameFixtureRoot() / "encoded").string(),
          "Folder where extracted JPEG frames will be written");
ABSL_FLAG(std::string, decoded_folder,
          (tools::DefaultFrameFixtureRoot() / "decoded").string(),
          "Folder where decoded PNG frames will be written");

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_log_path).empty()) {
    std::cerr << "--log_path is required\n";
    return 1;
  }

  const auto frames = tools::ExtractJpegLog(
      absl::GetFlag(FLAGS_log_path), absl::GetFlag(FLAGS_encoded_folder),
      absl::GetFlag(FLAGS_decoded_folder));
  std::cout << "Extracted " << frames.size() << " JPEG frame(s)\n";
  return 0;
}
