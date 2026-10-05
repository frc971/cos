add_library(yolo_inference)
target_sources(yolo_inference PRIVATE
  yolo_node.cc
)
target_link_libraries(yolo_inference PUBLIC
  cos::nvjpeg_decode_node
  cos::control_loop
  cos::thread_pool
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

add_library(gamepiece INTERFACE)
target_link_libraries(gamepiece INTERFACE
  yolo_inference
)
