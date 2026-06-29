add_cos_unit_test(localization_unit_tests
  networktable_sender_test.cc
  position_sender_test.cc
  position_solver_test.cc
  solver_nodes_test.cc
)
target_link_libraries(localization_unit_tests PRIVATE
  networktable_sender
  square_solver_node
  multi_tag_solver_node
  unambiguous_solver_node
  loop_controller
)
