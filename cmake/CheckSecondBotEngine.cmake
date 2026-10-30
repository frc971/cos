if(NOT EXISTS "${ENGINE_PATH}" OR IS_DIRECTORY "${ENGINE_PATH}")
    message(FATAL_ERROR
        "Second-bot engine not found: ${ENGINE_PATH}. Restore models/ball_unet.engine from Git or run scripts/export_unet_engine.sh and configure -DSECOND_BOT_ENGINE=/path/to/ball_unet.engine.")
endif()

file(SIZE "${ENGINE_PATH}" ENGINE_SIZE)
if(ENGINE_SIZE EQUAL 0)
    message(FATAL_ERROR
        "Second-bot engine is empty: ${ENGINE_PATH}. Restore models/ball_unet.engine from Git or regenerate it with scripts/export_unet_engine.sh.")
endif()
