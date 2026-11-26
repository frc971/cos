# Vision-bot NetworkTables pathing

`vision_bot_main` uses the existing team-9972 NT4 client and runs pathing in a
separate 20 ms control loop with one context. The camera pipeline does not provide
the pathing pose: pathing uses the roboRIO's fused current pose. The pipeline is
enabled by default; `--nopathing` disables it. Construction validates configuration
and loads the navgrid before camera startup. It does not create another NT client.

The live navgrid MJPEG stream is enabled by default at
`http://vision-bot:4975/navgrid`. It shows red obstacles, the green grid route,
the blue fitted spline, magenta control waypoints, and a cyan target-velocity
arrow. Frames are limited to 10 Hz and drawn after NT command publication.
Use `--nonavgrid_streamer` to disable this visualization without disabling
pathing. `--nopathing` disables the entire pathing pipeline, including its stream.
The default port follows camera streams 4971–4974.

## Inputs from the roboRIO

| Topic | NT type | Meaning |
| --- | --- | --- |
| `/DriveState/Pose` | `struct:Pose2d` | Current fused robot pose |
| `/COS/Pathing/Goal` | `struct:Pose2d` | Desired field position and robot heading |

Both topics use WPILib `Pose2d` serialization. Positions are meters; rotations
use the WPILib struct representation. Both poses must use the navgrid's fixed
field coordinate frame. COS does not mirror the goal for alliance color. The
roboRIO is responsible for any alliance conversion before publishing.

Publish the robot pose as it changes. COS uses the latest available pose and goal
without an input age limit. To cancel pathing, unpublish the goal topic. Pausing
updates leaves the last goal active while the topic exists. COS returns zero
commands when either input is missing, and rejects malformed, wrong-type,
nonfinite, and out-of-field poses. For each plan,
COS treats mapped obstacle cells overlapping the robot's starting footprint as free. Collision
and robot-radius checks still apply to every other obstacle and the field
boundary.

## Command to the roboRIO

`/COS/Pathing/Command` is a `double[]` with exactly seven fields, published as a
single frame. Metadata and velocities must be read together, not from separate
topics.

| Index | Value | Units |
| --- | --- | --- |
| 0 | Target `vx`, field-relative | m/s |
| 1 | Target `vy`, field-relative | m/s |
| 2 | Target `omega`, positive counterclockwise | rad/s |
| 3 | Status code below | Integer encoded as a double |
| 4 | Pose input server timestamp | NT server microseconds |
| 5 | Goal input server timestamp | NT server microseconds |
| 6 | Command creation server timestamp | NT server microseconds |

Input timestamps identify the samples used for planning, not later heartbeats
received during the calculation. Creation time is converted from the COS local
NT clock using its server time offset. Unavailable input timestamps or an
unavailable server-time offset are represented by zero, never NaN.

| Code | Status | Meaning |
| --- | --- | --- |
| 0 | Inactive | Inputs have not been published, or orderly shutdown |
| 1 | Following | Valid translation and/or rotation command |
| 2 | ReachedGoal | Within both position and heading tolerances |
| 3 | InvalidInput | Invalid topic type, pose, or nonfinite output |
| 4 | StaleInput | Reserved for command protocol compatibility |
| 5 | NoPath | No grid route exists |
| 6 | FitFailed | No usable clear spline/lookahead could be produced |
| 7 | Disconnected | NT connection or server time synchronization is unavailable |

Only status **1** can have nonzero velocity components. All other statuses set
all three components to zero. COS rechecks topic availability, topic types, and connection state immediately
before publication. Status changes are logged once
per transition. An orderly stop joins the pathing worker and publishes a final
zero command; a disconnected client cannot deliver that zero until reconnecting.

The future rio consumer must validate the array length, finite values, status,
input/creation timestamps, and whether the command belongs to its current goal.
It must stop/brake on any invalid or stale command, on cancellation, and whenever
the robot is disabled. It must enforce an independent **250 ms command watchdog**
using server-time age as well as local receipt age; receiving a cached command
does not make it fresh. Keep the consumer's maximum command age no longer than
COS's `pathing_command_timeout_ms`. These checks and actual drivetrain actuation
are not implemented in this repository.

## Following and configuration

Each cycle fits a new spline from the current pose. The follower picks a point
approximately 0.5 meters ahead along its samples, shortening the lookahead to
keep the direct translation segment clear of obstacles. Speed depends on
remaining path distance and is limited to 2 m/s by default. It is additionally capped by
the lookahead distance divided by the command hold interval, so nominal motion
during that interval cannot pass the selected point. Within 5 cm of the goal,
translation stops. The follower uses the shortest angular error and caps rotation
at 1 rad/s by default. `ReachedGoal` also requires heading to be within 3 degrees.
Set the rotation cap to zero to disable rotation and use position alone for goal
completion. Live pathing uses a circular robot radius of **0.19 m**
(0.38 m diameter). Route search, spline collision checks, field boundaries,
and follower lookahead all enforce this clearance from blocked navgrid cells.
The stream shows cells whose centers cannot accommodate the footprint in red.
The navgrid should describe obstacles without pre-inflating for the robot;
existing baked-in margins remain additional clearance. Set
`--pathing_robot_radius_m` to override the radius (zero uses a point footprint).

| Flag | Default |
| --- | --- |
| `--pathing` / `--nopathing` | Enabled |
| `--pathing_robot_radius_m` | 0.19 meters |
| `--navgrid_streamer` / `--nonavgrid_streamer` | Enabled with pathing |
| `--navgrid_streamer_port` | 4975 |
| `--navgrid_streamer_field_image` | `/root/constants/assets/field.png` |
| `--pathing_navgrid` | `/root/constants/assets/navgrid.json` |
| `--pathing_pose_topic` | `/DriveState/Pose` |
| `--pathing_goal_topic` | `/COS/Pathing/Goal` |
| `--pathing_command_topic` | `/COS/Pathing/Command` |
| `--pathing_period_ms` | 20 |
| `--pathing_input_timeout_ms` | 250 (deprecated, ignored) |
| `--pathing_command_timeout_ms` | 250 |
| `--pathing_lookahead_m` | 0.5 |
| `--pathing_max_translation_mps` | 2.0 |
| `--pathing_max_rotation_radps` | 1.0 |
| `--pathing_turn_penalty` | 0.3 meters/radian |

The C++ `NetworkTablesPathingSettings` also exposes the existing motion gains,
position/heading tolerances, and spline fitting budgets. The spline settings'
turn penalty controls route selection. `NetworkTablesPathing` accepts an NT
instance for isolated integration work and timestamps commands with `nt::Now()`.

This change supplies the COS producer only. It does not change roboRIO code,
add driver controls, or activate a drivetrain.
