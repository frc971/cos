add_library(gamepiece)
target_sources(gamepiece
  PRIVATE
    yolo_node.cc
)
target_link_libraries(gamepiece PUBLIC
  nvjpeg_decode_node
  loop_controller
  yolo
  nlohmann_json::nlohmann_json
  wpimath::wpimath
  ${OpenCV_LIBS}
  absl::check
  absl::log
)
target_include_directories(gamepiece PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)
