add_library(cos.dijkstra_planner)
add_library(cos::dijkstra_planner ALIAS cos.dijkstra_planner)
target_sources(cos.dijkstra_planner
  PRIVATE dijkstra_planner.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_planner.h
)
target_link_libraries(cos.dijkstra_planner PUBLIC
  cos::navgrid_streamer_node wpimath
)

add_library(cos.dijkstra_path_node)
add_library(cos::dijkstra_path_node ALIAS cos.dijkstra_path_node)
target_sources(cos.dijkstra_path_node
  PRIVATE dijkstra_path_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_path_node.h
)
target_link_libraries(cos.dijkstra_path_node PUBLIC
  cos::control_loop cos::dijkstra_planner
)

add_library(cos.dijkstra_spline_node)
add_library(cos::dijkstra_spline_node ALIAS cos.dijkstra_spline_node)
target_sources(cos.dijkstra_spline_node
  PRIVATE dijkstra_spline_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_spline_node.h
)
target_link_libraries(cos.dijkstra_spline_node PUBLIC cos::dijkstra_path_node)
