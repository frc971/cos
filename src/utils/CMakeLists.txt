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
)
target_include_directories(constants_from_json PUBLIC
  ${CMAKE_SOURCE_DIR}
)
