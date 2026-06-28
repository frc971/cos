add_library(square_solver_node)
target_sources(square_solver_node
  PRIVATE
    square_solver_node.cc
)
target_link_libraries(square_solver_node PUBLIC
  ${OpenCV_LIBS}
  wpimath::wpimath
  apriltag::apriltag
  absl::log
  constants_from_json
  camera_constants
  camera_utils
  transform
)
target_include_directories(square_solver_node PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(multi_tag_solver_node)
target_sources(multi_tag_solver_node
  PRIVATE
    multi_tag_solver_node.cc
)
target_link_libraries(multi_tag_solver_node PUBLIC
  ${OpenCV_LIBS}
  wpimath::wpimath
  apriltag::apriltag
  absl::log
  constants_from_json
  camera_constants
  camera_utils
  transform
  square_solver_node
)
target_include_directories(multi_tag_solver_node PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(unambiguous_solver_node)
target_sources(unambiguous_solver_node
  PRIVATE
    unambiguous_solver_node.cc
)
target_link_libraries(unambiguous_solver_node PUBLIC
  wpimath::wpimath
  apriltag::apriltag
  absl::check
  camera_constants
  multi_tag_solver_node
  transform
)
target_include_directories(unambiguous_solver_node PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(networktable_sender)
target_sources(networktable_sender
  PRIVATE
    networktable_sender.cc
)
target_link_libraries(networktable_sender PUBLIC
  ntcore::ntcore
  wpilibc::wpilibc
  wpimath::wpimath
  absl::log
)
target_include_directories(networktable_sender PUBLIC
  ${CMAKE_SOURCE_DIR}
)
