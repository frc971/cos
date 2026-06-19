include(${CMAKE_CURRENT_LIST_DIR}/TargetUtils.cmake)

get_all_targets(ALL_PROJECT_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(dev-orin
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying build/bin and build/lib to root@dev-orin:/root..."
    
    COMMAND ssh root@dev-orin "mkdir -p /root"
    COMMAND scp -r ${CMAKE_BINARY_DIR}/bin root@dev-orin:/root/
    COMMAND scp -r ${CMAKE_BINARY_DIR}/lib root@dev-orin:/root/
    
    COMMENT "Uploading folders to dev-orin..."
    VERBATIM
)
