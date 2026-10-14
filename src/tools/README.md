To calibrate intrinsics from recorded frames, first export ChArUco detections
for every frame in the directory, then sample usable detections.
On the device (use `build/tools/` for local build paths):

```sh
./tools/calib_helper --camera_folder=/path/to/frames --detections_output_path=detections.json
./tools/intrinsics_calibrate_disk --detections_path=detections.json --intrinsics_output_path=intrinsics.json
```

The helper processes the directory's frames one at a time and writes all
results, including frames without enough corners, directly to JSON.
The JSON records image dimensions and an array of detections containing
filenames, ChArUco corners and IDs, image points, and object points. The helper
does not run calibration.
`intrinsics_calibrate_disk` walks the recording in 0.5-second steps until it
collects 50 usable detections. If the first pass falls short, it continues from
the beginning offset by 0.25 seconds, keeping the first pass's detections and
skipping duplicates. If both passes fall short, calibration reports an error.
Timestamp filename stems must be in seconds. `--max_detections` changes the
capture count; `--max_detections=0` uses all usable results.
The saved JSON can be reused without decoding frames or running detection
again. For live calibration, run
`intrinsics_calibrate --config_path=/path/to/config.json`,
press Enter to capture each frame, and type `q` then Enter to calibrate and quit.
