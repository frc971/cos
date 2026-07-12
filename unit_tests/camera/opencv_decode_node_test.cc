#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "camera/opencv_decode_node.h"
#include "gtest/gtest.h"

namespace {

auto ReadBytes(const std::filesystem::path& path) -> std::vector<char> {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

auto EncodedFixturePath() -> std::filesystem::path {
  return std::filesystem::path(COS_SOURCE_DIR) / "unit_tests" / "testdata" /
         "jpeg_frames" / "encoded" / "20.283378.jpg";
}

TEST(OpenCVDecodeNodeTest, ImplementsDecodeNodeInterface) {
  EXPECT_TRUE((std::is_base_of_v<camera::IDecodeNode, camera::OpenCVDecodeNode>));
  EXPECT_FALSE(std::is_copy_constructible_v<camera::OpenCVDecodeNode>);
  EXPECT_FALSE(std::is_move_constructible_v<camera::OpenCVDecodeNode>);
}

TEST(OpenCVDecodeNodeTest, DecodesRealJpegFixture) {
  const std::filesystem::path fixture_path = EncodedFixturePath();
  const std::vector<char> jpeg_bytes = ReadBytes(fixture_path);
  ASSERT_GT(jpeg_bytes.size(), 0U) << "Missing JPEG fixture: " << fixture_path;

  auto jpeg_buffer = std::make_shared<camera::JpegBuffer>(jpeg_bytes.size());
  std::memcpy(jpeg_buffer->ptr(), jpeg_bytes.data(), jpeg_bytes.size());

  auto decoded_promise = std::make_shared<
      std::promise<std::shared_ptr<camera::DecodedJpegNvBuffer>>>();
  std::future<std::shared_ptr<camera::DecodedJpegNvBuffer>> decoded_future =
      decoded_promise->get_future();

  camera::OpenCVDecodeNode decoder("opencv_decode_node_test");
  decoder.RegisterCallback(
      [decoded_promise](std::shared_ptr<camera::DecodedJpegNvBuffer> decoded,
                        control_loops::MetaDataList,
                        std::shared_ptr<control_loops::Context>) {
        decoded_promise->set_value(std::move(decoded));
      });

  decoder.Decode(jpeg_buffer, {{.camera_idx = 0, .timestamp = 20283378}},
                 nullptr);

  ASSERT_EQ(decoded_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const std::shared_ptr<camera::DecodedJpegNvBuffer> decoded =
      decoded_future.get();
  ASSERT_NE(decoded, nullptr);
  ASSERT_NE(decoded->buffer, nullptr);
  ASSERT_GT(decoded->buffer->n_planes, 0U);
  EXPECT_GT(decoded->buffer->planes[0].fmt.width, 0U);
  EXPECT_GT(decoded->buffer->planes[0].fmt.height, 0U);
  EXPECT_GT(decoded->buffer->planes[0].bytesused, 0U);
}

TEST(OpenCVDecodeNodeTest, RunsNullTerminationThroughCallbacks) {
  camera::OpenCVDecodeNode decoder("opencv_decode_node_null_test");

  auto decoded_promise = std::make_shared<
      std::promise<std::shared_ptr<camera::DecodedJpegNvBuffer>>>();
  std::future<std::shared_ptr<camera::DecodedJpegNvBuffer>> decoded_future =
      decoded_promise->get_future();

  decoder.RegisterCallback(
      [decoded_promise](std::shared_ptr<camera::DecodedJpegNvBuffer> decoded,
                        control_loops::MetaDataList,
                        std::shared_ptr<control_loops::Context>) {
        decoded_promise->set_value(std::move(decoded));
      });

  decoder.Decode(nullptr, {}, nullptr);

  ASSERT_EQ(decoded_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(decoded_future.get(), nullptr);
}

}  // namespace
