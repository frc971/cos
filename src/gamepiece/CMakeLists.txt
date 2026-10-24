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

add_library(image_utils)
add_library(cos::image_utils ALIAS image_utils)
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

add_library(hsv_cluster_tracker_node hsv_cluster_tracker_node.cc)
target_link_libraries(hsv_cluster_tracker_node PUBLIC
  opencv_core wpimath cos::control_loop PRIVATE
  image_utils cos::cv_geometry_utils opencv_imgproc
  cos::nvjpeg_decode_node cos::thread_pool)
target_include_directories(hsv_cluster_tracker_node PUBLIC ${CMAKE_SOURCE_DIR}/src)
add_library(cos::hsv_cluster_tracker_node ALIAS hsv_cluster_tracker_node)

add_library(gamepiece INTERFACE)
target_link_libraries(gamepiece INTERFACE
  yolo_inference
  hsv_cluster_tracker_node
)
