include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(dev-orin
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying binaries, configs, and models to root@dev-orin:/root..."
    
    COMMAND ssh root@dev-orin "mkdir -p /root/models"
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum "${CMAKE_SOURCE_DIR}/models/" root@dev-orin:/root/models/

    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tests/ root@dev-orin:/root/tests/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/main/ root@dev-orin:/root/main/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/examples/ root@dev-orin:/root/examples/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tools/ root@dev-orin:/root/tools/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/lib/ root@dev-orin:/root/lib/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_SOURCE_DIR}/constants/ root@dev-orin:/root/constants/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_SOURCE_DIR}/systemd/dev-orin.service root@dev-orin:/etc/systemd/system
    
    COMMENT "Uploading folders to dev-orin..."
    USES_TERMINAL
    VERBATIM
)
