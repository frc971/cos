#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tools/jpeg_frame_tools.h"

ABSL_FLAG(std::string, decoded_folder,
          (tools::DefaultFrameFixtureRoot() / "decoded").string(),
          "Folder containing decoded image frames");
ABSL_FLAG(std::string, encoded_folder,
          (tools::DefaultFrameFixtureRoot() / "encoded").string(),
          "Folder where encoded JPEG frames will be written");
ABSL_FLAG(int, jpeg_quality, 95, "JPEG quality from 1 to 100");

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  const auto frames = tools::EncodeDecodedDirectory(
      absl::GetFlag(FLAGS_decoded_folder), absl::GetFlag(FLAGS_encoded_folder),
      absl::GetFlag(FLAGS_jpeg_quality));
  std::cout << "Encoded " << frames.size() << " image frame(s)\n";
  return 0;
}
