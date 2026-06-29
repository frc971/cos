add_cos_unit_test(camera_unit_tests
  camera_constants_test.cc
  nvjpeg_decode_node_test.cc
  uvc_camera_node_test.cc
)
target_link_libraries(camera_unit_tests PRIVATE
  camera_utils
  camera_constants
  constants_from_json
  nvjpeg_decode_node
  uvc_camera_node
)
target_compile_definitions(camera_unit_tests PRIVATE
  COS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
