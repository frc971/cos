add_executable(localization_example example.cc)
target_link_libraries(localization_example PRIVATE
  absl::flags
  absl::flags_parse
  absl::log
  absl::log_initialize
  absl::check
  nlohmann_json::nlohmann_json
  uvc_camera_node
  nvjpeg_decode_node
  loop_controller
  jpeg_buffer_streamer_node
  gpu_apriltag_detector_node
  opencv_apriltag_detector_node
  unambiguous_solver_node
  networktable_sender
  stop
)
