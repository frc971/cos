add_cos_unit_test(streamer_unit_tests
  jpeg_buffer_streamer_node_test.cc
)
target_link_libraries(streamer_unit_tests PRIVATE
  jpeg_buffer_streamer_node
)
