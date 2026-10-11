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

add_library(image_utils)
target_sources(image_utils PRIVATE
  image_utils.cc
)
target_link_libraries(image_utils PUBLIC
  opencv_core
  opencv_imgproc
  opencv_cudaarithm
  opencv_cudaimgproc
  wpimath
)
target_include_directories(image_utils PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(hsv_projector_node)
target_sources(hsv_projector_node PRIVATE
  hsv_projector_node.cc
)
target_link_libraries(hsv_projector_node PUBLIC
  cos::control_loop
  opencv_core
)
target_link_libraries(hsv_projector_node PRIVATE
  cos::nvjpeg_decode_node
  wpimath
)
target_include_directories(hsv_projector_node PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece INTERFACE)
target_link_libraries(gamepiece INTERFACE
  yolo_inference
  hsv_projector_node
)
