#include "tools/jpeg_frame_tools.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gtest/gtest.h"

namespace {

auto FixtureRoot() -> std::filesystem::path {
  return std::filesystem::path(COS_SOURCE_DIR) /
         tools::DefaultFrameFixtureRoot();
}

auto ArtifactRoot() -> std::filesystem::path {
  return std::filesystem::path(COS_BINARY_DIR) / "test_artifacts" /
         "jpeg_frames";
}

auto ReadBytes(const std::filesystem::path& path) -> std::vector<char> {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

auto LowercaseExtension(const std::filesystem::path& path) -> std::string {
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return extension;
}

auto ListFilesWithExtensions(const std::filesystem::path& folder,
                             const std::vector<std::string>& extensions)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string extension = LowercaseExtension(entry.path());
    if (std::ranges::find(extensions, extension) != extensions.end()) {
      files.push_back(entry.path());
    }
  }
  std::ranges::sort(files);
  return files;
}

auto EncodedFixtures() -> std::vector<std::filesystem::path> {
  return ListFilesWithExtensions(FixtureRoot() / "encoded", {".jpg", ".jpeg"});
}

auto DecodedFixtures() -> std::vector<std::filesystem::path> {
  return ListFilesWithExtensions(FixtureRoot() / "decoded", {".png"});
}

TEST(JpegFrameToolsTest, ReadsRealEncodedAndDecodedFixtureImages) {
  const auto encoded_fixtures = EncodedFixtures();
  const auto decoded_fixtures = DecodedFixtures();
  ASSERT_GT(encoded_fixtures.size(), 0)
      << "Expected real encoded frame fixtures in "
      << (FixtureRoot() / "encoded");
  ASSERT_GT(decoded_fixtures.size(), 0)
      << "Expected real decoded frame fixtures in "
      << (FixtureRoot() / "decoded");
  ASSERT_EQ(encoded_fixtures.size(), decoded_fixtures.size());

  const cv::Mat encoded =
      cv::imread(encoded_fixtures.front().string(), cv::IMREAD_COLOR);
  const cv::Mat decoded =
      cv::imread(decoded_fixtures.front().string(), cv::IMREAD_COLOR);

  ASSERT_FALSE(encoded.empty());
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(encoded.rows, decoded.rows);
  EXPECT_EQ(encoded.cols, decoded.cols);
}

TEST(JpegFrameToolsTest, DecodesRealJpegFixturesToDisk) {
  const std::filesystem::path root = FixtureRoot();
  const std::filesystem::path output = ArtifactRoot() / "decoded";
  std::filesystem::remove_all(output);
  const auto encoded_fixtures = EncodedFixtures();
  ASSERT_GT(encoded_fixtures.size(), 0)
      << "Expected real encoded frame fixtures in " << (root / "encoded");

  const auto frames = tools::DecodeJpegDirectory(root / "encoded", output);

  ASSERT_EQ(frames.size(), encoded_fixtures.size());
  for (const auto& frame : frames) {
    EXPECT_TRUE(std::filesystem::exists(frame.decoded_path));
    EXPECT_FALSE(cv::imread(frame.decoded_path.string()).empty());
  }
  std::cout << "Decoded frame proof written to " << output << '\n';
}

TEST(JpegFrameToolsTest, EncodesRealDecodedFixturesToDisk) {
  const std::filesystem::path root = FixtureRoot();
  const std::filesystem::path output =
      std::filesystem::path(::testing::TempDir()) / "encoded_frames";
  const auto decoded_fixtures = DecodedFixtures();
  ASSERT_GT(decoded_fixtures.size(), 0)
      << "Expected real decoded frame fixtures in " << (root / "decoded");

  const auto frames =
      tools::EncodeDecodedDirectory(root / "decoded", output, 90);

  ASSERT_EQ(frames.size(), decoded_fixtures.size());
  for (const auto& frame : frames) {
    EXPECT_TRUE(std::filesystem::exists(frame.encoded_path));
    EXPECT_FALSE(cv::imread(frame.encoded_path.string()).empty());
  }
}

TEST(JpegFrameToolsTest, ExtractsJpegsFromByteLogAndDecodesThem) {
  const std::filesystem::path temp =
      std::filesystem::path(::testing::TempDir());
  const std::filesystem::path log_path = temp / "jpeg_frames.log";

  const auto encoded_fixtures = EncodedFixtures();
  ASSERT_GT(encoded_fixtures.size(), 0)
      << "Expected real encoded frame fixtures in "
      << (FixtureRoot() / "encoded");
  const std::vector<char> jpeg = ReadBytes(encoded_fixtures.front());
  {
    std::ofstream output(log_path, std::ios::binary);
    output << "prefix";
    output.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
    output << "between";
    output.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
    output << "suffix";
  }

  const auto frames = tools::ExtractJpegLog(log_path, temp / "encoded_log",
                                            temp / "decoded_log");

  ASSERT_EQ(frames.size(), 2);
  EXPECT_TRUE(
      std::filesystem::exists(temp / "encoded_log" / "frame_000000.jpg"));
  EXPECT_TRUE(
      std::filesystem::exists(temp / "decoded_log" / "frame_000001.png"));
  EXPECT_FALSE(
      cv::imread((temp / "decoded_log" / "frame_000001.png").string()).empty());
}

TEST(JpegFrameToolsTest, ExtractsJpegEndingAtEndOfLog) {
  const std::filesystem::path temp =
      std::filesystem::path(::testing::TempDir());
  const std::filesystem::path log_path = temp / "jpeg_at_eof.log";

  const auto encoded_fixtures = EncodedFixtures();
  ASSERT_GT(encoded_fixtures.size(), 0)
      << "Expected real encoded frame fixtures in "
      << (FixtureRoot() / "encoded");
  const std::vector<char> jpeg = ReadBytes(encoded_fixtures.front());
  {
    std::ofstream output(log_path, std::ios::binary);
    output << "prefix";
    output.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
  }

  const auto frames = tools::ExtractJpegLog(log_path, temp / "encoded_at_eof",
                                            temp / "decoded_at_eof");

  ASSERT_EQ(frames.size(), 1);
  EXPECT_TRUE(
      std::filesystem::exists(temp / "encoded_at_eof" / "frame_000000.jpg"));
  EXPECT_TRUE(
      std::filesystem::exists(temp / "decoded_at_eof" / "frame_000000.png"));
}

TEST(JpegFrameToolsTest, RejectsEmptyFrameDirectories) {
  const std::filesystem::path temp =
      std::filesystem::path(::testing::TempDir());
  const std::filesystem::path empty_encoded = temp / "empty_encoded";
  const std::filesystem::path empty_decoded = temp / "empty_decoded";
  std::filesystem::create_directories(empty_encoded);
  std::filesystem::create_directories(empty_decoded);

  EXPECT_THROW(
      tools::DecodeJpegDirectory(empty_encoded, temp / "decoded_output"),
      std::runtime_error);
  EXPECT_THROW(
      tools::EncodeDecodedDirectory(empty_decoded, temp / "encoded_output", 90),
      std::runtime_error);
}

TEST(JpegFrameToolsTest, RejectsLogsWithoutJpegFrames) {
  const std::filesystem::path temp =
      std::filesystem::path(::testing::TempDir());
  const std::filesystem::path log_path = temp / "empty.log";
  {
    std::ofstream output(log_path, std::ios::binary);
    output << "not a jpeg";
  }

  EXPECT_THROW(
      tools::ExtractJpegLog(log_path, temp / "encoded", temp / "decoded"),
      std::runtime_error);
}

}  // namespace
