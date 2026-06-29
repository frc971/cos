#include <memory>
#include <cstring>
#include <type_traits>

#include "gtest/gtest.h"
#include "streamer/jpeg_buffer_streamer_node.h"

namespace {

TEST(JpegBufferStreamerNodeTest, IsConcreteClass) {
  EXPECT_FALSE(std::is_abstract_v<streamer::JpegBufferStreamerNode>);
}

TEST(JpegBufferStreamerNodeTest, PublishesBufferToConfiguredPath) {
  streamer::JpegBufferStreamerNode streamer("/stream", 50991);
  auto buffer = std::make_shared<camera::JpegBuffer>(4);
  std::memcpy(buffer->ptr(), "jpeg", 4);

  EXPECT_NO_THROW(streamer.Stream(buffer));
}

}  // namespace
