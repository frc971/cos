add_library(cos.pathing_messages INTERFACE)
add_library(cos::pathing_messages ALIAS cos.pathing_messages)
target_sources(cos.pathing_messages INTERFACE FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src
  FILES command_message.h pose_message.h spline_path_message.h
)
target_link_libraries(cos.pathing_messages INTERFACE cos::control_loop wpimath)

add_library(cos.dijkstra_planner)
add_library(cos::dijkstra_planner ALIAS cos.dijkstra_planner)
target_sources(cos.dijkstra_planner
  PRIVATE dijkstra_planner.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_planner.h
)
target_link_libraries(cos.dijkstra_planner PUBLIC
  cos::navgrid_message wpimath
)

add_library(cos.dijkstra_spline_node)
add_library(cos::dijkstra_spline_node ALIAS cos.dijkstra_spline_node)
target_sources(cos.dijkstra_spline_node
  PRIVATE dijkstra_spline_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_spline_node.h
)
target_link_libraries(cos.dijkstra_spline_node PUBLIC
  cos::pathing_messages cos::dijkstra_planner
  PRIVATE cos::pathing_input_message cos::polyline_message
)

add_library(cos.navgrid_source_node)
add_library(cos::navgrid_source_node ALIAS cos.navgrid_source_node)
target_sources(cos.navgrid_source_node
  PRIVATE navgrid_source_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES navgrid_source_node.h
)
target_link_libraries(cos.navgrid_source_node PUBLIC
  cos::control_loop cos::navgrid_message
)

add_library(cos.pathing_setup)
add_library(cos::pathing_setup ALIAS cos.pathing_setup)
target_sources(cos.pathing_setup
  PRIVATE pathing_setup.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES pathing_setup.h
)
target_link_libraries(cos.pathing_setup
  PUBLIC cos::dijkstra_spline_node cos::networktables_command_streamer_node
  PRIVATE cos::json_utils cos::navgrid_source_node cos::navgrid_streamer_node
    cos::robot_pose_receiver_node
)
