add_library(cos.nvidia_apriltag_detector_node)
add_library(cos::nvidia_apriltag_detector_node ALIAS cos.nvidia_apriltag_detector_node)
target_sources(cos.nvidia_apriltag_detector_node
  PRIVATE
    nvidia_apriltag_detector_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      nvidia_apriltag_detector_node.h
      tag_detections.h
)

add_library(cos.cpu_apriltag_detector_node)
add_library(cos::cpu_apriltag_detector_node ALIAS cos.cpu_apriltag_detector_node)
target_sources(cos.cpu_apriltag_detector_node
  PRIVATE
    cpu_apriltag_detector_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      cpu_apriltag_detector_node.h
      tag_detections.h
)
target_link_libraries(cos.cpu_apriltag_detector_node PUBLIC
  cos::logging_types
  apriltag
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  absl::log
)
target_link_libraries(cos.nvidia_apriltag_detector_node
  PUBLIC
    cos::logging_types
    ${OpenCV_LIBS}
    vpi
    cos::nvjpeg_fd_decode_node
    cos::nvjpeg_decode_node
    nlohmann_json::nlohmann_json
  PRIVATE
    cos::cuda_utils

)
