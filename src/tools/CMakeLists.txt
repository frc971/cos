add_executable(uvc_logger)
target_sources(uvc_logger
  PRIVATE
    uvc_logger.cc
)
target_link_libraries(uvc_logger PRIVATE
  jpeg_buffer_streamer_node
  uvc_camera_node
  stop
  loop_controller
  absl::flags_parse
  nvjpeg
  nvjpeg_decode_node
  JPEG::JPEG
  ${OpenCV_LIBS}
)

add_library(jpeg_frame_tools)
target_sources(jpeg_frame_tools
  PRIVATE
    jpeg_frame_tools.cc
)
target_link_libraries(jpeg_frame_tools PUBLIC
  ${OpenCV_LIBS}
)
target_include_directories(jpeg_frame_tools PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_executable(jpeg_extract_log)
target_sources(jpeg_extract_log
  PRIVATE
    jpeg_extract_log.cc
)
target_link_libraries(jpeg_extract_log PRIVATE
  jpeg_frame_tools
  absl::flags_parse
)

add_executable(jpeg_decode_frames)
target_sources(jpeg_decode_frames
  PRIVATE
    jpeg_decode_frames.cc
)
target_link_libraries(jpeg_decode_frames PRIVATE
  jpeg_frame_tools
  absl::flags_parse
)

add_executable(jpeg_encode_frames)
target_sources(jpeg_encode_frames
  PRIVATE
    jpeg_encode_frames.cc
)
target_link_libraries(jpeg_encode_frames PRIVATE
  jpeg_frame_tools
  absl::flags_parse
)
