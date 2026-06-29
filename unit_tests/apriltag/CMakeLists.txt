add_cos_unit_test(apriltag_unit_tests
  apriltag_detector_node_test.cc
)
target_link_libraries(apriltag_unit_tests PRIVATE
  opencv_apriltag_detector_node
  gpu_apriltag_detector_node
  absl::flags
)
