add_library(cos.jpeg_buffer_streamer_node)
add_library(cos::jpeg_buffer_streamer_node ALIAS cos.jpeg_buffer_streamer_node)
target_sources(cos.jpeg_buffer_streamer_node
  PRIVATE
    jpeg_buffer_streamer_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer_streamer_node.h
)
target_link_libraries(cos.jpeg_buffer_streamer_node PUBLIC
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  absl::log
  absl::log_initialize
)
