#include "tools/jpeg_frame_tools.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace tools {
namespace {

auto LowercaseExtension(const std::filesystem::path& path) -> std::string {
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return extension;
}

auto IsJpeg(const std::filesystem::path& path) -> bool {
  const std::string extension = LowercaseExtension(path);
  return extension == ".jpg" || extension == ".jpeg";
}

auto IsDecodedImage(const std::filesystem::path& path) -> bool {
  const std::string extension = LowercaseExtension(path);
  return extension == ".png" || extension == ".bmp" || extension == ".ppm" ||
         extension == ".pgm" || IsJpeg(path);
}

auto SortedFiles(const std::filesystem::path& folder,
                 bool (*predicate)(const std::filesystem::path&))
    -> std::vector<std::filesystem::path> {
  if (!std::filesystem::is_directory(folder)) {
    throw std::runtime_error("Input folder does not exist: " + folder.string());
  }

  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (entry.is_regular_file() && predicate(entry.path())) {
      files.push_back(entry.path());
    }
  }
  std::ranges::sort(files);
  return files;
}

auto FrameName(size_t index, std::string_view extension) -> std::string {
  std::ostringstream name;
  name << "frame_" << std::setw(6) << std::setfill('0') << index << extension;
  return name.str();
}

auto ReadFile(const std::filesystem::path& path) -> std::vector<unsigned char> {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open input file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output.good()) {
    throw std::runtime_error("Failed to write output file: " + path.string());
  }
}

void WriteDecodedImage(const std::filesystem::path& path,
                       const cv::Mat& decoded) {
  if (decoded.empty()) {
    throw std::runtime_error("Refusing to write an empty decoded image: " +
                             path.string());
  }
  if (!cv::imwrite(path.string(), decoded)) {
    throw std::runtime_error("Failed to write decoded image: " + path.string());
  }
}

}  // namespace

auto DefaultFrameFixtureRoot() -> std::filesystem::path {
  return std::filesystem::path("unit_tests") / "testdata" / "jpeg_frames";
}

auto DecodeJpegDirectory(const std::filesystem::path& encoded_folder,
                         const std::filesystem::path& decoded_folder)
    -> std::vector<FramePair> {
  std::filesystem::create_directories(decoded_folder);

  std::vector<FramePair> pairs;
  for (const auto& encoded_path : SortedFiles(encoded_folder, IsJpeg)) {
    cv::Mat decoded = cv::imread(encoded_path.string(), cv::IMREAD_COLOR);
    if (decoded.empty()) {
      throw std::runtime_error("Failed to decode JPEG: " +
                               encoded_path.string());
    }

    const std::filesystem::path decoded_path =
        decoded_folder / (encoded_path.stem().string() + ".png");
    WriteDecodedImage(decoded_path, decoded);
    pairs.push_back(
        FramePair{.encoded_path = encoded_path, .decoded_path = decoded_path});
  }
  if (pairs.empty()) {
    throw std::runtime_error("No JPEG frames found in: " +
                             encoded_folder.string());
  }
  return pairs;
}

auto EncodeDecodedDirectory(const std::filesystem::path& decoded_folder,
                            const std::filesystem::path& encoded_folder,
                            int jpeg_quality) -> std::vector<FramePair> {
  if (jpeg_quality < 1 || jpeg_quality > 100) {
    throw std::runtime_error("JPEG quality must be between 1 and 100");
  }

  std::filesystem::create_directories(encoded_folder);

  std::vector<FramePair> pairs;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
  for (const auto& decoded_path : SortedFiles(decoded_folder, IsDecodedImage)) {
    cv::Mat decoded = cv::imread(decoded_path.string(), cv::IMREAD_COLOR);
    if (decoded.empty()) {
      throw std::runtime_error("Failed to read decoded image: " +
                               decoded_path.string());
    }

    const std::filesystem::path encoded_path =
        encoded_folder / (decoded_path.stem().string() + ".jpg");
    if (!cv::imwrite(encoded_path.string(), decoded, params)) {
      throw std::runtime_error("Failed to write encoded JPEG: " +
                               encoded_path.string());
    }
    pairs.push_back(
        FramePair{.encoded_path = encoded_path, .decoded_path = decoded_path});
  }
  if (pairs.empty()) {
    throw std::runtime_error("No decoded image frames found in: " +
                             decoded_folder.string());
  }
  return pairs;
}

auto ExtractJpegLog(const std::filesystem::path& log_path,
                    const std::filesystem::path& encoded_folder,
                    const std::filesystem::path& decoded_folder)
    -> std::vector<FramePair> {
  std::filesystem::create_directories(encoded_folder);
  std::filesystem::create_directories(decoded_folder);

  const std::vector<unsigned char> log = ReadFile(log_path);
  std::vector<FramePair> pairs;
  size_t frame_index = 0;
  size_t offset = 0;
  while (offset + 1 < log.size()) {
    auto start = std::find(log.begin() + static_cast<std::ptrdiff_t>(offset),
                           log.end(), 0xFF);
    while (start != log.end() && std::next(start) != log.end() &&
           *std::next(start) != 0xD8) {
      start = std::find(std::next(start), log.end(), 0xFF);
    }
    if (start == log.end() || std::next(start) == log.end()) {
      break;
    }

    auto end = std::next(start, 2);
    bool found_end = false;
    while (end != log.end()) {
      if (*std::prev(end) == 0xFF && *end == 0xD9) {
        ++end;
        found_end = true;
        break;
      }
      ++end;
    }
    if (!found_end) {
      break;
    }

    std::vector<unsigned char> encoded(start, end);
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (decoded.empty()) {
      offset = static_cast<size_t>(std::distance(log.begin(), end));
      continue;
    }

    const std::filesystem::path encoded_path =
        encoded_folder / FrameName(frame_index, ".jpg");
    const std::filesystem::path decoded_path =
        decoded_folder / FrameName(frame_index, ".png");
    WriteBytes(encoded_path, encoded);
    WriteDecodedImage(decoded_path, decoded);
    pairs.push_back(
        FramePair{.encoded_path = encoded_path, .decoded_path = decoded_path});

    ++frame_index;
    offset = static_cast<size_t>(std::distance(log.begin(), end));
  }

  if (pairs.empty()) {
    throw std::runtime_error("No JPEG frames found in log: " +
                             log_path.string());
  }
  return pairs;
}

}  // namespace tools
