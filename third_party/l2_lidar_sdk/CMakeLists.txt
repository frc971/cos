find_package(Threads REQUIRED)

set(unilidar_sdk_archive
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunilidar_sdk2.a")
if(NOT EXISTS "${unilidar_sdk_archive}")
    message(FATAL_ERROR
        "The Unitree L2 LiDAR SDK has no archive for ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

add_library(unilidar_sdk2 STATIC IMPORTED GLOBAL)
set_target_properties(unilidar_sdk2 PROPERTIES
    IMPORTED_LOCATION "${unilidar_sdk_archive}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(unilidar_sdk2 INTERFACE Threads::Threads)
