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

#include <link.h>

#include "camera/nvjpeg_decode_node.h"
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

auto LinkedJpegLibraryPath() -> std::string {
  std::string path;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        std::string* path = static_cast<std::string*>(data);
        const std::string name =
            info->dlpi_name == nullptr ? "" : info->dlpi_name;
        if (name.find("libjpeg.so") != std::string::npos) {
          *path = name;
          return 1;
        }
        return 0;
      },
      &path);
  return path;
}

auto HasNvidiaJpegRuntime() -> bool {
  const std::string jpeg_path = LinkedJpegLibraryPath();
  return jpeg_path.find("/tegra/") != std::string::npos ||
         jpeg_path.find("nvidia") != std::string::npos;
}

TEST(DecodedJpegNvBufferTest, AllowsNullNativeBuffer) {
  camera::DecodedJpegNvBuffer buffer(nullptr);

  EXPECT_EQ(buffer.buffer, nullptr);
}

TEST(NvjpegDecodeNodeTest, ImplementsDecodedBufferNodeInterface) {
  EXPECT_TRUE(
      (std::is_base_of_v<INode<std::shared_ptr<camera::DecodedJpegNvBuffer>>,
                         camera::NvjpegDecodeNode>));
  EXPECT_FALSE(std::is_copy_constructible_v<camera::NvjpegDecodeNode>);
  EXPECT_FALSE(std::is_move_constructible_v<camera::NvjpegDecodeNode>);
}

TEST(NvjpegDecodeNodeTest, DecodesRealJpegFixture) {
  if (!HasNvidiaJpegRuntime()) {
    GTEST_SKIP() << "NvjpegDecodeNode requires NVIDIA's libjpeg runtime; "
                 << "linked libjpeg is " << LinkedJpegLibraryPath();
  }

  const std::filesystem::path fixture_path = EncodedFixturePath();
  const std::vector<char> jpeg_bytes = ReadBytes(fixture_path);
  ASSERT_GT(jpeg_bytes.size(), 0U) << "Missing JPEG fixture: " << fixture_path;

  auto jpeg_buffer = std::make_shared<camera::JpegBuffer>(jpeg_bytes.size());
  std::memcpy(jpeg_buffer->ptr(), jpeg_bytes.data(), jpeg_bytes.size());

  auto decoded_promise = std::make_shared<
      std::promise<std::shared_ptr<camera::DecodedJpegNvBuffer>>>();
  std::future<std::shared_ptr<camera::DecodedJpegNvBuffer>> decoded_future =
      decoded_promise->get_future();

  camera::NvjpegDecodeNode decoder("nvjpeg_decode_node_test");
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

}  // namespace
