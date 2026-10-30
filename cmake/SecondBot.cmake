include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

set(SECOND_BOT_ENGINE "${CMAKE_SOURCE_DIR}/models/ball_unet_rgb.engine" CACHE FILEPATH
    "TensorRT U-Net engine to deploy to second-bot")

add_custom_target(second-bot
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying binaries, configs, and U-Net engine to root@10.9.71.11:/root..."
    
    COMMAND ssh root@10.9.71.11 "mkdir -p /root/models"
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum "${CMAKE_SOURCE_DIR}/models/" root@10.9.71.11:/root/models/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum "${SECOND_BOT_ENGINE}" root@10.9.71.11:/root/models/ball_unet_rgb.engine

    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tests/ root@10.9.71.11:/root/tests/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/main/ root@10.9.71.11:/root/main/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/examples/ root@10.9.71.11:/root/examples/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/tools/ root@10.9.71.11:/root/tools/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_BINARY_DIR}/lib/ root@10.9.71.11:/root/lib/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial --delete ${CMAKE_SOURCE_DIR}/constants/ root@10.9.71.11:/root/constants/
    COMMAND rsync -avz --compress-level=9 --info=progress2 --checksum --partial ${CMAKE_SOURCE_DIR}/systemd/ root@10.9.71.11:/etc/systemd/system
    
    COMMENT "Uploading folders to 10.9.71.11..."
    USES_TERMINAL
    VERBATIM
)
