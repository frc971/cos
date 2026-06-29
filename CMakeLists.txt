cmake_minimum_required(VERSION 3.16) 
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

project(COS CXX C CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

option(ENABLE_CLANG_TIDY "Enable clang-tidy for static analysis" OFF)

add_compile_options(-march=armv8-a+simd)

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SYSROOT}/opt/nvidia/vpi3")

find_package(CUDAToolkit REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(wpilib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vpi REQUIRED)
find_package(libuvc REQUIRED)

add_subdirectory(third_party)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
enable_testing()
add_subdirectory(unit_tests)

include(cmake/DevOrin.cmake)
