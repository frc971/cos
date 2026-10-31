set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

add_executable(context_handoff_node_test context_handoff_node_test.cc)
target_link_libraries(context_handoff_node_test PRIVATE
  cos::control_loop GTest::gtest_main)
add_test(NAME context_handoff_node_test COMMAND context_handoff_node_test)

add_executable(yolo_node_test yolo_node_test.cc)
target_link_libraries(yolo_node_test PRIVATE
  yolo_inference
  GTest::gtest_main
)
add_test(NAME yolo_node_test COMMAND yolo_node_test)

add_executable(ball_pathing_logic_test ball_pathing_logic_test.cc)
target_link_libraries(ball_pathing_logic_test PRIVATE
  cos::ball_pathing_node
  GTest::gtest_main
)
add_test(NAME ball_pathing_logic_test COMMAND ball_pathing_logic_test)

add_executable(ball_pathing_node_test ball_pathing_node_test.cc)
target_link_libraries(ball_pathing_node_test PRIVATE
  cos::ball_pathing_node
  GTest::gtest_main
)
target_compile_definitions(ball_pathing_node_test PRIVATE
  COS_TEST_CONSTANTS_DIR="${PROJECT_SOURCE_DIR}/constants"
)
add_test(NAME ball_pathing_node_test COMMAND ball_pathing_node_test)

add_executable(unet_preprocess_test unet_preprocess_test.cc)
target_link_libraries(unet_preprocess_test PRIVATE
  cos::ball_pathing_node GTest::gtest_main)
add_test(NAME unet_preprocess_test COMMAND unet_preprocess_test)

add_executable(full_robot_gamepiece_pipeline_test
  full_robot_gamepiece_pipeline_test.cc)
target_link_libraries(full_robot_gamepiece_pipeline_test PRIVATE
  # OpenCV's JPEG writer needs the system libjpeg ABI. Put it before the
  # NVIDIA multimedia library, which exports overlapping JPEG symbols via
  # the AprilTag detector's FD input support.
  "-Wl,--push-state,--no-as-needed"
  JPEG::JPEG
  "-Wl,--pop-state"
  gamepiece
  cos::uvc_disk_camera_node
  cos::get_earliest_timestamp
  cos::rio_clock
  cos::stop
  cos::nvidia_apriltag_detector_node
  cos::unambiguous_solver_node
  cos::camera_config
  absl::flags_parse
  absl::log_initialize
)

add_executable(log_fields_test log_fields_test.cc)
target_link_libraries(log_fields_test PRIVATE cos::wpilog_writer
  cos::control_loop GTest::gtest_main GTest::gtest cos::localization_solver_common)
add_test(NAME log_fields_test COMMAND log_fields_test)

add_executable(networktables_fields_test networktables_fields_test.cc)
target_link_libraries(networktables_fields_test PRIVATE cos::wpilog_writer
  cos::control_loop GTest::gtest_main GTest::gtest cos::localization_solver_common
  cos::nvjpeg_decode_node cos::nvjpeg_fd_decode_node cos::thread_pool)
add_test(NAME networktables_fields_test COMMAND networktables_fields_test)

add_executable(wpilog_writer_test wpilog_writer_test.cc)
target_link_libraries(wpilog_writer_test PRIVATE cos::wpilog_writer
  cos::control_loop GTest::gtest_main)
add_test(NAME wpilog_writer_test COMMAND wpilog_writer_test)

add_executable(single_tag_rotation_test single_tag_rotation_test.cc)
target_link_libraries(single_tag_rotation_test PRIVATE
  GTest::gtest_main
  GTest::gtest
  cos::unambiguous_solver_node
  cos::control_loop
)
target_compile_definitions(single_tag_rotation_test PRIVATE
  COS_TEST_CONSTANTS_DIR="${PROJECT_SOURCE_DIR}/constants"
)
add_test(NAME single_tag_rotation_test COMMAND single_tag_rotation_test)

add_executable(nvjpeg_fd_decode_node_test nvjpeg_fd_decode_node_test.cc)
target_link_libraries(nvjpeg_fd_decode_node_test PRIVATE
  cos::nvjpeg_fd_decode_node
  cos::control_loop
  cos::thread_pool
  GTest::gtest_main
)
add_test(NAME nvjpeg_fd_decode_node_test COMMAND nvjpeg_fd_decode_node_test)

add_executable(test_thread_pool
  thread_pool_test.cc
)
target_link_libraries(test_thread_pool PRIVATE
  cos::thread_pool
  GTest::gtest_main
)

add_executable(test_cv_geometry
  cv_geometry_test.cc
)
target_link_libraries(test_cv_geometry PRIVATE
  cos::cv_geometry_utils
  GTest::gtest_main
)

add_executable(test_solver_common
  solver_common_test.cc
)
target_link_libraries(test_solver_common PRIVATE
  GTest::gtest_main
  GTest::gtest
  cos::localization_solver_common
)

add_executable(variance_calculator_node_test
  variance_calculator_node_test.cc
)
target_link_libraries(variance_calculator_node_test PRIVATE
  GTest::gtest_main
  GTest::gtest
  cos::variance_calculator_node
  cos::control_loop
)

add_executable(nvidia_apriltag_detector_node_test
  nvidia_apriltag_detector_node_test.cc
)
target_link_libraries(nvidia_apriltag_detector_node_test PRIVATE
  cos::jpeg_buffer_streamer_node
  cos::nvidia_apriltag_detector_node
  cos::nvjpeg_fd_decode_node
  cos::nvjpeg_decode_node
  cos::control_loop
  cos::thread_pool
  cos::stop
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  absl::flags_parse
  absl::log_initialize
  cos::uvc_camera_node
)

add_executable(timestamp_test
  timestamp_test.cc
)
target_link_libraries(timestamp_test PRIVATE
  cos::rio_clock
  wpiutil
  ntcore
)

add_executable(square_solve_node_test
  square_solver_node_test.cc
)
target_link_libraries(square_solve_node_test PRIVATE
  cos::square_solver_node
  cos::multi_tag_solver_node
  cos::nvjpeg_decode_node
  cos::nvidia_apriltag_detector_node
  cos::control_loop
  cos::thread_pool
  wpilibc
  ntcore
  cos::jpeg_buffer_streamer_node
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::camera_config
  absl::flags_parse
)

add_executable(unambiguous_solver_node_test
  unambiguous_solver_node_test.cc
)
target_link_libraries(unambiguous_solver_node_test PRIVATE
  cos::nvjpeg_fd_decode_node
  cos::nvidia_apriltag_detector_node
  cos::control_loop
  cos::thread_pool
  wpilibc
  ntcore
  cos::jpeg_buffer_streamer_node
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::camera_config
  absl::flags_parse
  cos::unambiguous_solver_node
  cos::simulation_position_sender_node
)

add_executable(ball_pathing_streamer_node_test ball_pathing_streamer_node_test.cc)
target_link_libraries(ball_pathing_streamer_node_test PRIVATE
  cos::ball_pathing_streamer_node opencv_calib3d GTest::gtest_main)
add_test(NAME ball_pathing_streamer_node_test COMMAND ball_pathing_streamer_node_test)

# Manual replay benchmark; requires camera recordings and the TensorRT engine.
add_executable(second_bot_fps_test second_bot_fps_test.cc)
target_link_libraries(second_bot_fps_test PRIVATE
  cos::uvc_disk_camera_node
  cos::get_earliest_timestamp
  cos::ball_pathing_streamer_node
  absl::check
  cos::ball_pathing_node
  cos::ball_velocity_goal_streamer_node
  cos::jpeg_buffer_streamer_node
  cos::uvc_camera_node
  cos::stop
  absl::flags_parse
  cos::nvjpeg_decode_node
  JPEG::JPEG
  cos::control_loop
  cos::wpilog_writer
  cos::thread_pool
  cos::cuda_utils
  wpilibc
  cos::nvidia_apriltag_detector_node
  cos::unambiguous_solver_node
  cos::variance_calculator_node
  cos::position_estimate_rio_streamer_node
  cos::jpeg_buffer_log_node
  cos::simulation_position_sender_node
  cos::connect_to_rio
  ${OpenCV_LIBS}
)
