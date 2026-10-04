add_library(yolo_inference)
target_sources(yolo_inference PRIVATE
  yolo.cc
)
target_link_libraries(yolo_inference PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  absl::check
  absl::log
  nvinfer
  nvinfer_plugin
)
target_include_directories(yolo_inference PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece_node)
target_sources(gamepiece_node PRIVATE
  gamepiece_node.cc
)
target_link_libraries(gamepiece_node PUBLIC
  cos::nvjpeg_decode_node
  cos::control_loop
  cos::thread_pool
  nlohmann_json::nlohmann_json
  wpimath
  opencv_core
  CUDA::cudart
  absl::check
  absl::log
)
target_include_directories(gamepiece_node PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece INTERFACE)
target_link_libraries(gamepiece INTERFACE
  gamepiece_node
  yolo_inference
)
