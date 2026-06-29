#include <type_traits>

#include "camera/nvjpeg_decode_node.h"
#include "gtest/gtest.h"

namespace {

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

}  // namespace
