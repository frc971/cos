add_library(cos.dijkstra_path_node)
add_library(cos::dijkstra_path_node ALIAS cos.dijkstra_path_node)
target_sources(cos.dijkstra_path_node
  PRIVATE dijkstra_path_node.cc
  PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES dijkstra_path_node.h
)
target_link_libraries(cos.dijkstra_path_node PUBLIC
  cos::control_loop cos::navgrid_streamer_node wpilibc
)
