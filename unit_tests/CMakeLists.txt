include(GoogleTest)

function(add_cos_unit_test target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE GTest::gtest_main)
  target_include_directories(${target} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )
  gtest_discover_tests(${target})
endfunction()

add_subdirectory(apriltag)
add_subdirectory(camera)
add_subdirectory(control_loops)
add_subdirectory(localization)
add_subdirectory(streamer)
add_subdirectory(logging)
add_subdirectory(tools)
