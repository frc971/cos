#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

struct FramePair {
  std::filesystem::path encoded_path;
  std::filesystem::path decoded_path;
};

auto DecodeJpegDirectory(const std::filesystem::path& encoded_folder,
                         const std::filesystem::path& decoded_folder)
    -> std::vector<FramePair>;

auto EncodeDecodedDirectory(const std::filesystem::path& decoded_folder,
                            const std::filesystem::path& encoded_folder,
                            int jpeg_quality) -> std::vector<FramePair>;

auto ExtractJpegLog(const std::filesystem::path& log_path,
                    const std::filesystem::path& encoded_folder,
                    const std::filesystem::path& decoded_folder)
    -> std::vector<FramePair>;

auto DefaultFrameFixtureRoot() -> std::filesystem::path;

}  // namespace tools
