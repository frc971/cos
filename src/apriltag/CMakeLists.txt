add_library(opencv_apriltag_detector_node)
target_sources(opencv_apriltag_detector_node
  PRIVATE
    opencv_apriltag_detector_node.cc
)
target_link_libraries(opencv_apriltag_detector_node PUBLIC
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
  absl::log
  constants_from_json
  nvjpeg_decode_node
)

add_library(gpu_apriltag_detector_node)
target_sources(gpu_apriltag_detector_node
  PRIVATE
    gpu_apriltag_detector_node.cc
)
target_link_libraries(gpu_apriltag_detector_node PUBLIC
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
  absl::log
  absl::status
  971apriltag
  constants_from_json
  nvjpeg_decode_node
)
target_include_directories(gpu_apriltag_detector_node PUBLIC
  ${CMAKE_SOURCE_DIR}
)
