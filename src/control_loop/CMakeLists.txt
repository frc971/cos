add_library(cos.control_loop)
add_library(cos::control_loop ALIAS cos.control_loop)
target_sources(cos.control_loop
  PRIVATE
    context_handoff_node.cc
    control_loop.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      context.h
      context_handoff_node.h
      control_loop.h
      message.h
      node.h
      timed_node.h
)
target_compile_options(cos.control_loop PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(cos.control_loop PRIVATE
  absl::check
  absl::log
  cos::timer
  cos::wpilog_writer
)

find_package(Threads REQUIRED)

add_library(cos.thread_pool)
add_library(cos::thread_pool ALIAS cos.thread_pool)
target_sources(cos.thread_pool
  PRIVATE
    thread_pool.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      thread_pool.h
)
target_link_libraries(cos.thread_pool PUBLIC
  Threads::Threads
  absl::check
)

add_library(cos.timer)
add_library(cos::timer ALIAS cos.timer)
target_sources(cos.timer
  PRIVATE
    timer.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      timer.h
)

add_library(cos.rio_clock)
add_library(cos::rio_clock ALIAS cos.rio_clock)
target_sources(cos.rio_clock
  PRIVATE
    rio_clock.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      rio_clock.h
)
target_link_libraries(cos.rio_clock PRIVATE
  wpiutil
  ntcore
  absl::log
)

add_library(cos.connect_to_rio)
add_library(cos::connect_to_rio ALIAS cos.connect_to_rio)
target_sources(cos.connect_to_rio
  PRIVATE
    connect_to_rio.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      connect_to_rio.h
)
target_link_libraries(cos.connect_to_rio PRIVATE
  absl::log
  cos::stop
  ntcore
  wpiutil
)
