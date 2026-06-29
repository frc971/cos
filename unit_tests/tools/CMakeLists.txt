add_cos_unit_test(tools_unit_tests
  jpeg_frame_tools_test.cc
)
target_link_libraries(tools_unit_tests PRIVATE
  jpeg_frame_tools
  ${OpenCV_LIBS}
)
target_compile_definitions(tools_unit_tests PRIVATE
  COS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  COS_BINARY_DIR="${CMAKE_BINARY_DIR}"
)
