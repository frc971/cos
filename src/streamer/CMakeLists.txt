add_library(cos.navgrid_message navgrid_message.cc)
add_library(cos::navgrid_message ALIAS cos.navgrid_message)
target_sources(cos.navgrid_message
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES navgrid_message.h
)
target_link_libraries(cos.navgrid_message
  PUBLIC cos::control_loop opencv_core
  PRIVATE nlohmann_json::nlohmann_json
)

add_library(cos.polyline_message INTERFACE)
add_library(cos::polyline_message ALIAS cos.polyline_message)
target_sources(cos.polyline_message INTERFACE FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src
  FILES polyline_message.h
)
target_link_libraries(cos.polyline_message INTERFACE cos::control_loop opencv_core)

add_library(cos.navgrid_streamer_node)
find_library(TURBOJPEG_LIBRARY NAMES turbojpeg REQUIRED)
# The Jetson image need not include TurboJPEG. Deployment copies build/lib,
# and executables already search it through their $ORIGIN/../lib RPATH.
file(COPY "${TURBOJPEG_LIBRARY}" DESTINATION "${CMAKE_BINARY_DIR}/lib"
  FOLLOW_SYMLINK_CHAIN)
add_library(cos::navgrid_streamer_node ALIAS cos.navgrid_streamer_node)
target_sources(cos.navgrid_streamer_node
  PRIVATE
    navgrid_streamer_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      navgrid_streamer_node.h
)
target_link_libraries(cos.navgrid_streamer_node PUBLIC
  cos::control_loop
  absl::log
  cos::navgrid_message
  cos::polyline_message
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  PRIVATE cos::pathing_messages cos::pathing_input_message
    ${TURBOJPEG_LIBRARY}
)

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

add_library(cos.networktables_command_streamer_node)
add_library(cos::networktables_command_streamer_node ALIAS cos.networktables_command_streamer_node)
target_sources(cos.networktables_command_streamer_node
  PRIVATE networktables_command_streamer_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES networktables_command_streamer_node.h
)
target_link_libraries(cos.networktables_command_streamer_node
  PUBLIC cos::pathing_messages ntcore
  PRIVATE absl::log
)
