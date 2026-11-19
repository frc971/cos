# Control loop

Each iteration starts with a shared `Context` that holds messages for that iteration. The loop runs dependency nodes before its registered callbacks. Publishers invoke their downstream callbacks with the same context, so each subscriber can look up the inputs available at that point. If asynchronous work retains the context, its slot stays occupied until that work releases it.

## Scheduling and messages

- [control_loop.h](control_loop.h) declares `ControlLoop`. Its API configures the loop period and maximum number of in-flight contexts, registers nodes and callbacks, and controls startup, shutdown, and logging.
- [control_loop.cc](control_loop.cc) validates that each channel has one publisher with one type and that regular nodes' dependencies match a published type. It attaches subscriber callbacks to the corresponding publishers and starts a `std::jthread` that creates a fresh context for each available slot, runs the dependency callbacks and then the loop callbacks, and sleeps for the configured period between passes. It also records optional latency and throughput metrics, writes contexts to WPILog when they are destroyed, and flushes the writer on shutdown.
- [context.h](context.h) defines the shared per-iteration state. `ContextInternal` carries the iteration ID, start time, loop pointer, stop token, and metrics flag. It stores the first message inserted for each channel in a mutex-protected map and provides typed lookups with borrowed or shared ownership.
- [message.h](message.h) defines the `IMessage` interface for runtime type and size queries. `ValueMessage<T>` wraps plain payloads for storage in a context. `MessageDescriptor` names a channel, lists its allowed message types, and carries WPILog registration metadata for typed publications.
- [node.h](node.h) defines the `INode` contract for declaring input and output channels, creating a callback, and registering downstream callbacks. A subscriber can run once per declared dependency, so it checks which inputs are present before using them. A publisher inserts a null message for a declared output it could not produce.

## Handoffs, workers, and timing

- [context_handoff_node.h](context_handoff_node.h) declares `ContextHandoffNode`, an adapter that exposes a source node's publications in another loop. Register it as a dependency node in the receiving loop so its outputs are available before that loop's regular callbacks run. The purpose of this node is to allow different control loops to use the same resources (e.g. a gamepiece control loop that wants to run slower than the main localization loop)
- [context_handoff_node.cc](context_handoff_node.cc) registers a callback on the source node that saves the latest non-null shared message per channel under a mutex. The receiving callback waits on a condition variable for a source callback, moves the pending messages into its own context, and invokes downstream callbacks. Sharing message pointers allows the source context to be released independently.
- [thread_pool.h](thread_pool.h) declares a fixed-size `ThreadPool` and its task submission, size, and shutdown APIs. It defines a priority queue that orders pending tasks by highest numeric priority first.
- [thread_pool.cc](thread_pool.cc) starts the workers and uses a condition variable to wake them for queued tasks. Workers take tasks from the queue under a mutex and execute them after releasing it. Shutdown wakes the workers, lets them drain queued tasks, and joins them.
- [timed_node.h](timed_node.h) defines the `ITimedNode` interface for enabling timing on a named channel. It also defines `LatencyMessage`, which stores a duration in seconds and registers that field for WPILog.
- [timer.h](timer.h) declares `Timer`, a utility for measuring elapsed time from construction. Its API exposes the start time and returns elapsed time in seconds.
- [timer.cc](timer.cc) captures the start time with `std::chrono::steady_clock` and subtracts it from the current time on each `Stop()` call.

## roboRIO communication and logs

- [connect_to_rio.h](connect_to_rio.h) declares NetworkTables client and host startup helpers, run log directory helpers, and the image logging switch.
- [connect_to_rio.cc](connect_to_rio.cc) starts the NetworkTables client for the configured team and waits for a connection and clock synchronization or a stop request. It can also start a local server. It creates the first available numbered run log directory and publishes its path on `/COS/LogName`. It reads Driver Station enable transitions to keep image logging active through the autonomous-to-teleoperated gap and stop it after teleop disables.
- [rio_clock.h](rio_clock.h) declares `RioClock`, the shared API for getting synchronized NetworkTables server time or elapsed simulation time in seconds.
- [rio_clock.cc](rio_clock.cc) adds the signed NetworkTables server offset to the local WPILib timestamp and converts microseconds to seconds. It returns `NaN` while the offset is unavailable. In simulation, it measures elapsed time from a resettable steady-clock start.

## Build targets

- [CMakeLists.txt](CMakeLists.txt) groups the sources and public headers into five libraries and links their dependencies. Other targets can link them through the `cos::control_loop`, `cos::thread_pool`, `cos::timer`, `cos::rio_clock`, and `cos::connect_to_rio` aliases.
