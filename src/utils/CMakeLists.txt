add_library(stop)
target_sources(stop
  PRIVATE
    stop.cc
)
target_link_libraries(stop PUBLIC
  absl::log
)

add_library(constants_from_json)
target_sources(constants_from_json
  PRIVATE
    constants_from_json.cc
)
target_link_libraries(constants_from_json PUBLIC
  nlohmann_json::nlohmann_json
  971apriltag
  ${OpenCV_LIBS}
  Eigen3::Eigen
  wpimath::wpimath
)
target_include_directories(constants_from_json PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(camera_utils)
target_sources(camera_utils
  PRIVATE
    camera_utils.cc
)
target_link_libraries(camera_utils PUBLIC
  nlohmann_json::nlohmann_json
  absl::log
)
target_include_directories(camera_utils PUBLIC
  ${CMAKE_SOURCE_DIR}
)

add_library(transform)
target_sources(transform
  PRIVATE
    transform.cc
)
target_link_libraries(transform PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  wpimath::wpimath
)
target_include_directories(transform PUBLIC
  ${CMAKE_SOURCE_DIR}
)
