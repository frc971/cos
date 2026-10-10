#include "camera/nvjpeg_fd_decode_node.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

#include "control_loop/context.h"
#include "tests/valid_jpeg_fixture.h"

namespace {

// These tests use NVIDIA's decoder and must run on the Jetson.
class NvjpegFdDecodeNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_.RegisterCallback([this](const control_loop::Context&) -> void {
      ++callback_count_;
      completed_.set_value();
    });
  }

  void TearDown() override { thread_pool_.Shutdown(); }

  auto Decode(std::unique_ptr<camera::JpegBuffer> input)
      -> control_loop::Context {
    completed_ = std::promise<void>{};
    auto done = completed_.get_future();
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, ++id_);
    context->SetMessage("jpeg_buffer:test_camera", std::move(input));
    node_.CreateCallback()(context);
    EXPECT_EQ(done.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(callback_count_.load(), id_);
    EXPECT_TRUE(context->Exists("decoded"));
    return context;
  }

  auto Decode(const std::vector<unsigned char>& bytes) -> control_loop::Context {
    auto input = std::make_unique<camera::JpegBuffer>(bytes.size(), 123.0);
    if (!bytes.empty()) {
      std::memcpy(input->ptr, bytes.data(), bytes.size());
    }
    return Decode(std::move(input));
  }

  void ExpectDropped(const std::vector<unsigned char>& bytes) {
    auto context = Decode(bytes);
    EXPECT_EQ(context->GetMessage<camera::DecodedJpegFdBuffer>("decoded"),
              nullptr);
  }

  control_loop::ThreadPool thread_pool_{1};
  camera::NvjpegFdDecodeNode node_{"jpeg_buffer:test_camera", "decoded",
                                  thread_pool_};
  std::promise<void> completed_;
  std::atomic<uint64_t> callback_count_{0};
  uint64_t id_ = 0;
};

TEST_F(NvjpegFdDecodeNodeTest, DropsNullInput) {
  auto context = Decode(std::unique_ptr<camera::JpegBuffer>{});
  EXPECT_EQ(context->GetMessage<camera::DecodedJpegFdBuffer>("decoded"), nullptr);
}

TEST_F(NvjpegFdDecodeNodeTest, DropsNullData) {
  auto context = Decode(std::make_unique<camera::JpegBuffer>());
  EXPECT_EQ(context->GetMessage<camera::DecodedJpegFdBuffer>("decoded"), nullptr);
}

TEST_F(NvjpegFdDecodeNodeTest, DropsEmptyInput) {
  ExpectDropped({});
}

TEST_F(NvjpegFdDecodeNodeTest, DropsOneByteInput) {
  ExpectDropped({0xFF});
}

TEST_F(NvjpegFdDecodeNodeTest, DropsObservedInvalidStartMarker) {
  ExpectDropped({0xFF, 0xC0, 0x00, 0x11});
}

TEST_F(NvjpegFdDecodeNodeTest, DropsInvalidFirstByte) {
  ExpectDropped({0x00, 0xD8});
}

TEST_F(NvjpegFdDecodeNodeTest, DecodesValidFramesAroundRejectedFrame) {
  const std::vector<unsigned char> jpeg(std::begin(kValidJpeg),
                                       std::end(kValidJpeg));
  for (int i = 0; i < 2; ++i) {
    auto context = Decode(jpeg);
    auto* output = context->GetMessage<camera::DecodedJpegFdBuffer>("decoded");
    ASSERT_NE(output, nullptr);
    EXPECT_GE(output->fd, 0);
    EXPECT_EQ(output->width, 128);
    EXPECT_EQ(output->height, 128);
    if (i == 0) {
      ExpectDropped({0xFF, 0xC0});
    }
  }
}

TEST_F(NvjpegFdDecodeNodeTest, RecoversFromInvalidScanComponentRepeatedly) {
  const std::vector<unsigned char> jpeg(std::begin(kValidJpeg),
                                       std::end(kValidJpeg));
  auto corrupt = jpeg;
  const std::array<unsigned char, 2> sos{0xFF, 0xDA};
  auto marker = std::ranges::search(corrupt, sos).begin();
  ASSERT_NE(marker, corrupt.end());
  ASSERT_GE(corrupt.end() - marker, 6);
  // SOS marker, length (2 bytes), component count, first component ID.
  marker[5] = 255;
  for (int i = 0; i < 50; ++i) {
    ExpectDropped(corrupt);
    auto context = Decode(jpeg);
    auto* output = context->GetMessage<camera::DecodedJpegFdBuffer>("decoded");
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->width, 128);
    EXPECT_EQ(output->height, 128);
  }
}

TEST_F(NvjpegFdDecodeNodeTest, RecoversFromTruncatedHeader) {
  ExpectDropped({0xFF, 0xD8, 0xFF, 0xD9});
  auto context = Decode(std::vector<unsigned char>(std::begin(kValidJpeg),
                                                  std::end(kValidJpeg)));
  EXPECT_NE(context->GetMessage<camera::DecodedJpegFdBuffer>("decoded"), nullptr);
}

}  // namespace
