add_library(cos.jpeg_buffer_log_node)
add_library(cos::jpeg_buffer_log_node ALIAS cos.jpeg_buffer_log_node)
target_sources(cos.jpeg_buffer_log_node
  PRIVATE
    jpeg_buffer_log_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer_log_node.h
)
target_link_libraries(cos.jpeg_buffer_log_node PRIVATE
  absl::log
)

add_library(cos.wpilog_writer)
add_library(cos::wpilog_writer ALIAS cos.wpilog_writer)
target_sources(cos.wpilog_writer
  PRIVATE wpilog_writer.cc
  PUBLIC FILE_SET HEADERS BASE_DIRS ${PROJECT_SOURCE_DIR}/src
  FILES log_registration.h wpilog_writer.h
)
target_link_libraries(cos.wpilog_writer PUBLIC wpiutil wpimath ntcore)
