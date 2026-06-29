add_cos_unit_test(control_loops_unit_tests
  loop_controller_test.cc
)
target_link_libraries(control_loops_unit_tests PRIVATE
  loop_controller
)
