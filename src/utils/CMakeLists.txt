add_library(cos.stop)
add_library(cos::stop ALIAS cos.stop)
target_sources(cos.stop
  PRIVATE
    stop.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      stop.h
)
target_link_libraries(cos.stop PUBLIC
  absl::log
)

add_library(cos.cuda_utils)
add_library(cos::cuda_utils ALIAS cos.cuda_utils)
target_sources(cos.cuda_utils
  PRIVATE
    cuda_utils.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      cuda_utils.h
)
target_link_libraries(cos.cuda_utils PUBLIC
  CUDA::cudart
  absl::check
)

add_library(cos.json_utils)
add_library(cos::json_utils ALIAS cos.json_utils)
target_sources(cos.json_utils
  PRIVATE
    json.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      json.h
)
target_link_libraries(cos.json_utils PUBLIC
  nlohmann_json::nlohmann_json
  absl::check
)

add_library(cos.cv_geometry_utils)
add_library(cos::cv_geometry_utils ALIAS cos.cv_geometry_utils)
target_sources(cos.cv_geometry_utils
  PRIVATE
    cv_geometry.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      cv_geometry.h
)
target_link_libraries(cos.cv_geometry_utils PUBLIC
  opencv_calib3d
  opencv_core
  Eigen3::Eigen
  wpimath
)
target_link_libraries(cos.cv_geometry_utils PRIVATE
  apriltag
)
