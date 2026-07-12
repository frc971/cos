add_library(yolo)
target_sources(yolo
  PRIVATE
    yolo.cc
)
target_link_libraries(yolo PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  absl::check
  absl::log
  nvinfer
  nvinfer_plugin
)
target_include_directories(yolo PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)
