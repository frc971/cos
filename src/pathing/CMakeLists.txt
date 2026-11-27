add_library(cos.pathing_messages INTERFACE)
add_library(cos::pathing_messages ALIAS cos.pathing_messages)
target_sources(cos.pathing_messages INTERFACE FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src
  FILES pose_message.h pathing_status.h spline_path_message.h
    published_command_message.h
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
  PRIVATE cos::polyline_message
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

add_library(cos.networktables_pathing)
add_library(cos::networktables_pathing ALIAS cos.networktables_pathing)
target_sources(cos.networktables_pathing
  PRIVATE networktables_pathing.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES networktables_pathing.h motion_settings.h
)
target_link_libraries(cos.networktables_pathing
  PUBLIC cos::dijkstra_spline_node cos::control_loop ntcore
  PRIVATE absl::log cos::navgrid_source_node cos::navgrid_streamer_node
    cos::robot_pose_node
)
