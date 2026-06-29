#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tools/jpeg_frame_tools.h"

ABSL_FLAG(std::string, encoded_folder,
          (tools::DefaultFrameFixtureRoot() / "encoded").string(),
          "Folder containing encoded JPEG frames");
ABSL_FLAG(std::string, decoded_folder,
          (tools::DefaultFrameFixtureRoot() / "decoded").string(),
          "Folder where decoded PNG frames will be written");

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  const auto frames = tools::DecodeJpegDirectory(
      absl::GetFlag(FLAGS_encoded_folder), absl::GetFlag(FLAGS_decoded_folder));
  std::cout << "Decoded " << frames.size() << " JPEG frame(s)\n";
  return 0;
}
