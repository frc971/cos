add_library(camera_constants)
target_sources(camera_constants
  PRIVATE
    camera_constants.cc
)
target_link_libraries(camera_constants PUBLIC
  nlohmann_json::nlohmann_json
  absl::log
  absl::check
  absl::flags
)
target_include_directories(camera_constants PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(uvc_camera_node)
target_sources(uvc_camera_node
  PRIVATE
    uvc_camera_node.cc
)
target_link_libraries(uvc_camera_node PRIVATE
  uvc
  absl::log
  absl::check
  nlohmann_json::nlohmann_json
  camera_constants
)

add_library(nvjpeg_decode_node)
target_sources(nvjpeg_decode_node
  PRIVATE
    nvjpeg_decode_node.cc
)
target_compile_options(nvjpeg_decode_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(nvjpeg_decode_node PUBLIC
  nvjpeg
  absl::log
)
