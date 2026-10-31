add_library(cos.lane_density)
add_library(cos::lane_density ALIAS cos.lane_density)
target_sources(cos.lane_density
  PRIVATE
    lane_density.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      lane_density.h
)
target_link_libraries(cos.lane_density PUBLIC
  cos::camera_config
)
target_link_libraries(cos.lane_density PRIVATE
  cos::image_utils
  cos::cv_geometry_utils
  opencv_calib3d
  absl::flags
  absl::log
)

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

add_library(cos.ball_pathing_node)
add_library(cos::ball_pathing_node ALIAS cos.ball_pathing_node)
set_target_properties(cos.ball_pathing_node PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
)
target_compile_options(cos.ball_pathing_node PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--sysroot=${CMAKE_SYSROOT}>
)
target_sources(cos.ball_pathing_node
  PRIVATE
    ball_pathing_node.cc
    ball_pathing_logic.cc
    unet_preprocess.cu
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      ball_pathing_node.h
      ball_pathing_logic.h
      unet_preprocess.h
)
target_link_libraries(cos.ball_pathing_node PUBLIC
  cos::camera_config
  cos::control_loop
  cos::logging_types
  cos::nvjpeg_decode_node
  cos::thread_pool
  opencv_core
  opencv_imgproc
  opencv_calib3d
  CUDA::cudart
  nvinfer
)
