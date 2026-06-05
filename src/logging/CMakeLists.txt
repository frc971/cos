add_library(png_image_log_node)
target_sources(png_image_log_node
  PRIVATE
    png_image_log_node.cc
)
target_link_libraries(png_image_log_node PRIVATE
  absl::log
)

