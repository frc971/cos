add_executable(path_camera_sim main.cc field_renderer.cc)
target_link_libraries(path_camera_sim PRIVATE
  absl::flags_parse
  absl::flags
  cos::camera_config
  EGL
  GLESv2
  opencv_core
  opencv_calib3d
  opencv_imgcodecs
  opencv_imgproc
  nlohmann_json::nlohmann_json
)
