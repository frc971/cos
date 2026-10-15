add_library(cos.uvc_camera_node)
add_library(cos::uvc_camera_node ALIAS cos.uvc_camera_node)
target_sources(cos.uvc_camera_node
  PRIVATE
    uvc_camera_config.cc
    uvc_camera_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer.h
      uvc_camera_config.h
      uvc_camera_node.h
)
target_link_libraries(cos.uvc_camera_node PRIVATE
  uvc
  absl::log
  nlohmann_json::nlohmann_json
  wpilibc
  cos::rio_clock
)

set(TEGRA_MMAPI_ROOT
  "${CMAKE_SYSROOT}/usr/src/debug/tegra-mmapi-samples/39.2.0-20260601141651"
  CACHE PATH "Path to the Jetson Multimedia API samples")
set(TEGRA_NVJPEG_HEADERS
  "${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-nvjpeg/1.14.0-r39.2.0/nv_headers"
  CACHE PATH "Path to NVIDIA's accelerated libjpeg headers")

set(TEGRA_LIBJPEG_SHIM "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers/libjpeg-9f")
file(MAKE_DIRECTORY "${TEGRA_LIBJPEG_SHIM}")
foreach(header jpeglib.h jmorecfg.h jerror.h)
  configure_file(
    "${TEGRA_NVJPEG_HEADERS}/${header}"
    "${TEGRA_LIBJPEG_SHIM}/${header}"
    COPYONLY
  )
endforeach()
configure_file(
  "${PROJECT_SOURCE_DIR}/third_party/tegra-mmapi/libjpeg-9f/jconfig.h"
  "${TEGRA_LIBJPEG_SHIM}/jconfig.h"
  COPYONLY
)

find_package(Threads REQUIRED)
find_library(NVMM_JPEG_LIBRARY NAMES nvmm_jpeg REQUIRED)
find_library(NVBUF_SURFACE_LIBRARY NAMES nvbufsurface REQUIRED)
find_library(NVBUF_TRANSFORM_LIBRARY NAMES nvbufsurftransform REQUIRED)

add_library(cos.jetson_mmjpeg STATIC)
add_library(cos::jetson_mmjpeg ALIAS cos.jetson_mmjpeg)
target_sources(cos.jetson_mmjpeg PRIVATE
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvElement.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvElementProfiler.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvBufSurface.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvBuffer.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvLogging.cpp"
)
target_include_directories(cos.jetson_mmjpeg
  PUBLIC
    "${TEGRA_MMAPI_ROOT}/include"
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers"
)
target_compile_definitions(cos.jetson_mmjpeg PRIVATE TEGRA_ACCELERATE)
target_compile_options(cos.jetson_mmjpeg PRIVATE -Wno-unused-parameter -Wno-extra)
target_link_libraries(cos.jetson_mmjpeg
  PUBLIC
    ${NVMM_JPEG_LIBRARY}
    ${NVBUF_SURFACE_LIBRARY}
    ${NVBUF_TRANSFORM_LIBRARY}
    CUDA::cuda_driver
    Threads::Threads
)

add_library(cos.nvjpeg_decode_node)
add_library(cos::nvjpeg_decode_node ALIAS cos.nvjpeg_decode_node)
target_sources(cos.nvjpeg_decode_node
  PRIVATE
    nvjpeg_decode_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer.h
      nvjpeg_decode_node.h
)
target_compile_options(cos.nvjpeg_decode_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(cos.nvjpeg_decode_node
  PUBLIC
    CUDA::cudart
    CUDA::nvjpeg
    cos::cuda_utils
    absl::log
  PRIVATE
    cos::timer
)

add_library(cos.nvjpeg_fd_decode_node)
add_library(cos::nvjpeg_fd_decode_node ALIAS cos.nvjpeg_fd_decode_node)
target_sources(cos.nvjpeg_fd_decode_node
  PRIVATE
    nvjpeg_fd_decode_node.cc
    safe_nvjpeg_decoder.c
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer.h
      nvjpeg_fd_decode_node.h
)
target_include_directories(cos.nvjpeg_fd_decode_node PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers"
)
target_compile_definitions(cos.nvjpeg_fd_decode_node PRIVATE TEGRA_ACCELERATE)
target_link_libraries(cos.nvjpeg_fd_decode_node
  PUBLIC
    cos::jetson_mmjpeg
    absl::log
  PRIVATE
    cos::timer
)

add_library(cos.cpu_decode_node)
add_library(cos::cpu_decode_node ALIAS cos.cpu_decode_node)
target_sources(cos.cpu_decode_node
  PRIVATE
    cpu_decode_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      cpu_decode_node.h
      jpeg_buffer.h
)
target_link_libraries(cos.cpu_decode_node PUBLIC
  JPEG::JPEG
  absl::log
)

add_library(cos.camera_config)
add_library(cos::camera_config ALIAS cos.camera_config)
target_sources(cos.camera_config
  PRIVATE
    camera_config.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      camera_config.h
)
target_link_libraries(cos.camera_config PUBLIC
  opencv_core
  wpimath
  cos::cv_geometry_utils
  absl::check
)
target_link_libraries(cos.camera_config PRIVATE
  nlohmann_json::nlohmann_json
)

add_library(cos.uvc_disk_camera_node)
add_library(cos::uvc_disk_camera_node ALIAS cos.uvc_disk_camera_node)
target_sources(cos.uvc_disk_camera_node
  PRIVATE
    uvc_disk_camera_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer.h
      uvc_disk_camera_node.h
)
target_link_libraries(cos.uvc_disk_camera_node PRIVATE
  absl::log
  cos::control_loop
  cos::rio_clock
  cos::stop
)

add_library(cos.get_earliest_timestamp)
add_library(cos::get_earliest_timestamp ALIAS cos.get_earliest_timestamp)
target_sources(cos.get_earliest_timestamp
  PRIVATE
    get_earliest_timestamp.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      get_earliest_timestamp.h
)
