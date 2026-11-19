include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(lidar-orin
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying build/bin and build/lib to root@lidar-orin:/root..."
    
    COMMAND ssh root@lidar-orin "mkdir -p /root"

    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tests/ root@lidar-orin:/root/tests/
    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/main/ root@lidar-orin:/root/main/
    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/examples/ root@lidar-orin:/root/examples/
    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tools/ root@lidar-orin:/root/tools/
    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/lib/ root@lidar-orin:/root/lib/
    COMMAND rsync -avz --no-times --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_SOURCE_DIR}/constants/ root@lidar-orin:/root/constants/
    
    COMMENT "Uploading folders to lidar-orin..."
    USES_TERMINAL
    VERBATIM
)
