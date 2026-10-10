cmake -B build -G Ninja -Wno-dev
cmake --build build

Camera configuration JSON stores extrinsic `rotation_x`, `rotation_y`, and
`rotation_z` in degrees and translations in meters. Rotations are converted to
radians immediately when loaded.

The robot programs record timestamped UVC JPEG frames by default under
`/cos/logs/logN/<camera-name>/`. Use `--nolog_images` to disable image logging.
Recording uses the boolean `/AdvantageKit/DriverStation/Enabled` topic. It starts
on the first enable, continues through the first disable (the auto-to-teleop gap)
and second enable, and stops on the second disable. This cycle repeats for the
next match. A NetworkTables disconnect or missing topic stops recording and
resets the cycle. The standalone `uvc_logger` tool records unconditionally when
given `--log_folder`.
