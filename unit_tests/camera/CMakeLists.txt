add_cos_unit_test(camera_unit_tests
  camera_constants_test.cc
  nvjpeg_decode_node_test.cc
  uvc_camera_node_test.cc
)
target_link_libraries(camera_unit_tests PRIVATE
  camera_constants
  nvjpeg_decode_node
  uvc_camera_node
)
