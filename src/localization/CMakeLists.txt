add_library(cos.localization_solver_common)
add_library(cos::localization_solver_common ALIAS cos.localization_solver_common)
target_sources(cos.localization_solver_common
  PRIVATE
    solver_common.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      position.h
      solver_common.h
)
target_link_libraries(cos.localization_solver_common PUBLIC
  ${OpenCV_LIBS}
  wpimath
  apriltag
  cos::nvidia_apriltag_detector_node
  nlohmann_json::nlohmann_json
)

add_library(cos.square_solver_node)
add_library(cos::square_solver_node ALIAS cos.square_solver_node)
target_sources(cos.square_solver_node
  PRIVATE
    square_solver_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      square_solver_node.h
)
target_link_libraries(cos.square_solver_node PUBLIC
  cos::localization_solver_common
  cos::camera_config
  cos::cv_geometry_utils
  cos::json_utils
)

add_library(cos.multi_tag_solver_node)
add_library(cos::multi_tag_solver_node ALIAS cos.multi_tag_solver_node)
target_sources(cos.multi_tag_solver_node
  PRIVATE
    multi_tag_solver_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      multi_tag_solver_node.h
)
target_link_libraries(cos.multi_tag_solver_node PUBLIC
  cos::localization_solver_common
  cos::square_solver_node
  cos::cv_geometry_utils
  cos::json_utils
  absl::log
)

add_library(cos.unambiguous_solver_node)
add_library(cos::unambiguous_solver_node ALIAS cos.unambiguous_solver_node)
target_sources(cos.unambiguous_solver_node
  PRIVATE
    unambiguous_solver_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      unambiguous_solver_node.h
)
target_link_libraries(cos.unambiguous_solver_node PUBLIC
  cos::localization_solver_common
  cos::multi_tag_solver_node
)
target_link_libraries(cos.unambiguous_solver_node PRIVATE
  absl::log
)

add_library(cos.variance_calculator_node)
add_library(cos::variance_calculator_node ALIAS cos.variance_calculator_node)
target_sources(cos.variance_calculator_node
  PRIVATE
    variance_calculator_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      variance_calculator_node.h
)
target_link_libraries(cos.variance_calculator_node PUBLIC
  cos::localization_solver_common
)
