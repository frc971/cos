add_executable(uvc_streamer)
target_sources(uvc_streamer
  PRIVATE
    uvc_streamer.cc
)
target_link_libraries(uvc_streamer PRIVATE
  jpeg_buffer_streamer_node
  uvc_camera_node
  stop
  absl::flags_parse
)

add_executable(uvc_logger)
target_sources(uvc_logger
  PRIVATE
    uvc_logger.cc
)
target_link_libraries(uvc_logger PRIVATE
  jpeg_buffer_streamer_node
  uvc_camera_node
  stop
  absl::flags_parse
  nvjpeg
  nvjpeg_decode_node
)
