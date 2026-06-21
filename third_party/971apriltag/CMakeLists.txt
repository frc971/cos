find_package(wpilib REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUB_COMPAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cub_compat")

add_library(971apriltag_dep)
target_sources(971apriltag_dep
    PRIVATE
    apriltag.c
)
set_target_properties(971apriltag_dep PROPERTIES LINKER_LANGUAGE C)

add_library(971apriltag_cuda_dep cuda.cc)
target_link_libraries(971apriltag_cuda_dep PUBLIC apriltag absl::log absl::check 971apriltag_dep CUDA::cudart)
target_include_directories(971apriltag_cuda_dep PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CUB_COMPAT_DIR}
)
set_source_files_properties(threshold.cc PROPERTIES LANGUAGE CUDA)

add_library(971apriltag_threshold_dep threshold.cc fatal_neon_threshold.cc)
target_link_libraries(971apriltag_threshold_dep PUBLIC apriltag absl::log absl::check 971apriltag_dep)
target_include_directories(971apriltag_threshold_dep PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CUB_COMPAT_DIR}
)

add_library(971apriltag
  apriltag.cc
  apriltag_detect.cc
  labeling_allegretti_2019_BKE.cc
  line_fit_filter.cc
  points.cc
)

target_link_libraries(971apriltag PUBLIC apriltag 971apriltag_cuda_dep 971apriltag_threshold_dep absl::log absl::check absl::status 971apriltag_dep)
target_include_directories(971apriltag PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CUB_COMPAT_DIR}
)
set_source_files_properties(labeling_allegretti_2019_BKE.cc apriltag_detect.cc apriltag.cc line_fit_filter.cc points.cc PROPERTIES LANGUAGE CUDA)

target_compile_options(971apriltag PRIVATE -w)
target_compile_options(971apriltag_dep PRIVATE -w)
target_compile_options(971apriltag_threshold_dep PRIVATE -w)
