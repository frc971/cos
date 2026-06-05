add_library(nvjpeg)
target_sources(nvjpeg
  PRIVATE
    NvBuffer.cpp
    NvElement.cpp
    NvElementProfiler.cpp
    NvJpegDecoder.cpp
    NvLogging.cpp
)
target_include_directories(nvjpeg PUBLIC
  /usr/src/jetson_multimedia_api/include
  /usr/src/jetson_multimedia_api/include/libjpeg-8b
)
