add_library(cos.jpeg_buffer_streamer_node)
add_library(cos::jpeg_buffer_streamer_node ALIAS cos.jpeg_buffer_streamer_node)
target_sources(cos.jpeg_buffer_streamer_node
  PRIVATE
    jpeg_buffer_streamer_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer_streamer_node.h
)
target_link_libraries(cos.jpeg_buffer_streamer_node PUBLIC
  cos::logging_types
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  absl::log
  absl::log_initialize
)

add_library(cos.position_estimate_rio_streamer_node)
add_library(cos::position_estimate_rio_streamer_node ALIAS cos.position_estimate_rio_streamer_node)
target_sources(cos.position_estimate_rio_streamer_node
  PRIVATE
    position_estimate_rio_streamer_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      position_estimate_rio_streamer_node.h
)
target_link_libraries(cos.position_estimate_rio_streamer_node PUBLIC
  absl::log
  wpilibc
  ntcore
)

add_library(cos.ball_velocity_goal_streamer_node)
add_library(cos::ball_velocity_goal_streamer_node ALIAS cos.ball_velocity_goal_streamer_node)
target_sources(cos.ball_velocity_goal_streamer_node
  PRIVATE
    ball_velocity_goal_streamer_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      ball_velocity_goal_streamer_node.h
)
target_link_libraries(cos.ball_velocity_goal_streamer_node PUBLIC
  cos::ball_pathing_node
  cos::control_loop
  ntcore
)
