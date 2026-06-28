if(ENABLE_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY
        clang-tidy;
        -checks=modernize-*,performance-*;
        -warnings-as-errors=*;
    )
endif()

include_directories(.)

add_subdirectory(utils)
add_subdirectory(control_loops)
add_subdirectory(apriltag)
add_subdirectory(camera)
add_subdirectory(streamer)
add_subdirectory(localization)
add_subdirectory(logging)
add_subdirectory(tools)
add_subdirectory(main)
add_subdirectory(examples)

set(CMAKE_CXX_CLANG_TIDY "")
