add_library(loop_controller)
target_sources(loop_controller
  PRIVATE
    loop_controller.cc
)
target_link_libraries(loop_controller PUBLIC
  absl::check
)
target_include_directories(loop_controller PUBLIC
  ${CMAKE_SOURCE_DIR}
)
