# Keep this adapter outside the upstream, Make-based SDK submodule.
set(RPLIDAR_SDK_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/rplidar_sdk"
    CACHE PATH "Path to the RPLIDAR SDK checkout")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The cos RPLIDAR SDK adapter currently supports Linux only")
endif()

if(NOT EXISTS "${RPLIDAR_SDK_SOURCE_DIR}/sdk/include/sl_lidar_driver.h")
  message(FATAL_ERROR
    "RPLIDAR SDK sources missing at ${RPLIDAR_SDK_SOURCE_DIR}. "
    "Run git submodule update --init --recursive, or set "
    "-DRPLIDAR_SDK_SOURCE_DIR=/path/to/rplidar_sdk.")
endif()

find_package(Threads REQUIRED)

# Mirrors sdk/Makefile at the pinned upstream revision.
set(_rplidar_sdk_sources
  sl_lidar_driver.cpp
  hal/thread.cpp
  sl_crc.cpp
  sl_serial_channel.cpp
  sl_lidarprotocol_codec.cpp
  sl_async_transceiver.cpp
  sl_tcp_channel.cpp
  sl_udp_channel.cpp
  dataunpacker/dataunpacker.cpp
  dataunpacker/unpacker/handler_capsules.cpp
  dataunpacker/unpacker/handler_hqnode.cpp
  dataunpacker/unpacker/handler_normalnode.cpp
  arch/linux/net_serial.cpp
  arch/linux/net_socket.cpp
  arch/linux/timer.cpp
)
list(TRANSFORM _rplidar_sdk_sources PREPEND "${RPLIDAR_SDK_SOURCE_DIR}/sdk/src/")

add_library(rplidar_sdk STATIC ${_rplidar_sdk_sources})
add_library(rplidar::sdk ALIAS rplidar_sdk)
set_target_properties(rplidar_sdk PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(rplidar_sdk PRIVATE -funsigned-char)
target_include_directories(rplidar_sdk SYSTEM PUBLIC
  "${RPLIDAR_SDK_SOURCE_DIR}/sdk/include"
)
target_include_directories(rplidar_sdk PRIVATE
  "${RPLIDAR_SDK_SOURCE_DIR}/sdk/src"
  "${RPLIDAR_SDK_SOURCE_DIR}/sdk/src/dataunpacker"
  "${RPLIDAR_SDK_SOURCE_DIR}/sdk/src/dataunpacker/unpacker"
)
target_link_libraries(rplidar_sdk PUBLIC Threads::Threads rt)
unset(_rplidar_sdk_sources)
