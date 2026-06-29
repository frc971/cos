# Targets
- Minimal holdup of the main localization loop
- Only interacts through the decode nodes, otherwise has no involvement
- Maintains a separate control loop with its own cycle time

# Implementation
- Use ../bos/src/gamepiece/ and ../bos/src/yolo/ especially as references, but make them node style
- A LoopController (a second one, separate from the one which runs localization) takes decoded frames from the nvjpeg decode nodes that are part of the localization loop and acts as a synchronizer there.
- When the loop begins, the LoopController calls its callbacks, among which there will be YOLO obj detect nodes. For now, what comes after YOLO is undecided and no callbacks will be registered on YOLO nodes YET (still make YOLO nodes node-styled for future implementation)
