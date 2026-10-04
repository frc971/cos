set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

add_subdirectory(path_camera_sim)

add_executable(uvc_logger)
target_sources(uvc_logger
  PRIVATE
    uvc_logger.cc
)
target_link_libraries(uvc_logger PRIVATE
  cos::jpeg_buffer_streamer_node
  cos::uvc_camera_node
  cos::stop
  absl::flags_parse
  cos::nvjpeg_decode_node
  JPEG::JPEG
  cos::control_loop
  cos::thread_pool
  cos::cuda_utils
  wpilibc
  cos::jpeg_buffer_log_node
  ${OpenCV_LIBS}
)

add_executable(compress_jpeg)
target_sources(compress_jpeg
  PRIVATE
    compress_jpeg.cc
)
target_link_libraries(compress_jpeg PRIVATE
  absl::flags_parse
  absl::check
  ${OpenCV_LIBS}
)

add_executable(jpeg_disk_replay)
target_sources(jpeg_disk_replay
  PRIVATE
    jpeg_disk_replay.cc
)
target_link_libraries(jpeg_disk_replay PRIVATE
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::jpeg_buffer_streamer_node
  cos::stop
  absl::flags_parse
  cos::control_loop
  cos::thread_pool
)

add_executable(intrinsics_calibrate)
target_sources(intrinsics_calibrate
  PRIVATE
    intrinsics_calibrate.cc
)
target_link_libraries(intrinsics_calibrate PRIVATE
  cos::uvc_camera_node
  cos::control_loop
  cos::stop
  absl::check
  absl::flags_parse
  absl::log_initialize
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  nlohmann_json::nlohmann_json
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_objdetect
  opencv_calib3d
)
target_include_directories(intrinsics_calibrate PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(calib_helper calib_helper.cc)
target_link_libraries(calib_helper PRIVATE
  cos::stop
  absl::flags_parse
  absl::log_initialize
  nlohmann_json::nlohmann_json
  opencv_core
  opencv_imgcodecs
  opencv_objdetect
)
target_include_directories(calib_helper PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(intrinsics_calibrate_disk intrinsics_calibrate_disk.cc)
target_link_libraries(intrinsics_calibrate_disk PRIVATE
  cos::stop
  absl::check
  absl::flags_parse
  absl::log_initialize
  nlohmann_json::nlohmann_json
  opencv_core
  opencv_objdetect
  opencv_calib3d
)
target_include_directories(intrinsics_calibrate_disk PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
