add_cos_unit_test(pipeline_unit_tests
  pipeline_test.cc
)
target_link_libraries(pipeline_unit_tests PRIVATE
  opencv_apriltag_detector_node
  nvjpeg_decode_node
  opencv_decode_node
  unambiguous_solver_node
  networktable_sender
  loop_controller
  uvc_camera_node
  disk_camera
  gamepiece
  datalog::datalog
)
target_compile_definitions(pipeline_unit_tests PRIVATE
  COS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  COS_BINARY_DIR="${CMAKE_BINARY_DIR}"
)
