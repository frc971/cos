add_subdirectory(abseil-cpp)
target_compile_options(absl_cord PRIVATE -Wno-maybe-uninitialized)
add_subdirectory(cpp-mjpeg-streamer)
add_subdirectory(json)
