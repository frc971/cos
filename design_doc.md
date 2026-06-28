# Design Goals:
- Minimal copy of non-trivial objects
- Structs are modular and contain only the information the consumer needs
  - If node A needs to communicate with node C, a direct connection should be made to node C instead of passing that information through C's connection with B
- The program can be safely terminated using ctrl+c or sigkill
- The same information is not processed twice
- Logging is easy
- Unit testing and integration testing are easy
- Nodes adhere to interfaces which make it easy to swap them out (e.g. GPUApriltagDetector can be swapped for OpencvApriltagDetector)
- Optional fields are avoided as much as possible
- Code is easily extensible
- std::shared_ptr is used as much as possible to allow flexible but safe access
- Everything is a node

# Implementation:
- An overarching `LoopController` object is defined and controls:
  - A stop token which determines whether another loop iteration will be run
  - A defined number of cameras, for which a callback is registered which takes every jpeg frame output by the cameras and places into a buffer owned by LoopController
  - An overall control loop
- The LoopController begins the loop by:
  - Copying the jpeg frames into a dedicated buffer that the cameras won't touch.
  - Creating a `std::shared_ptr` to a `Context` object. The sync object is defined as:
```
{
std::atomic<bool> stop_token;
weak_ptr<LoopController> controller;
~Context() {
    controller.wake_up()
}
}
```
  - The context object is then passed to every node that registers a callback. When all nodes are done, the reference count will naturally go to zero and the context object will be destroyed, thereby telling the controller to begin the next loop
  - The stop token can be activated by any node and will tell the controller not to begin the next loop (it is owned by the controller)
  - The `LoopController` calls its callbacks which expect jpeg frames. Some of these callbacks are for the mjpeg streamer, and the others are Decode nodes which will produce std::shared_ptr to a buffer. This buffer will be either a DecodedNvJPegBuffer or a cv::Mat depending on whether nvjpeg decoder or opencv decoder is used, so the detector nodes in the next step must be ready for both.
- The callbacks for the detector are called, and they pass on the tag_detection_t results to the solver node
- Every time the solver node's callback is called, it checks if it has received one result per camera yet. If it has, it does the solve and passes its output on to callbacks
- The sender node is awoken and sends the position estimate alongside any debug information
- Throughout this whole process, pieces of information which must be sent to a certain node but are useless for intermediate nodes, such as debug information, should be passed directly to the end node. DO NOT create bloated structs to pass this on through linear callbacks
