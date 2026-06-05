add_library(uvc_camera_node)
target_sources(uvc_camera_node
  PRIVATE
    uvc_camera_node.cc
)
target_link_libraries(uvc_camera_node PRIVATE
  uvc
  absl::log
)

add_library(nvjpeg_decode_node)
target_sources(nvjpeg_decode_node
  PRIVATE
    nvjpeg_decode_node.cc
)
target_link_libraries(nvjpeg_decode_node PRIVATE
  nvjpeg
  absl::log
)

