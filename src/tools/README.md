To calibrate intrinsics from recorded frames, first export ChArUco detections
for every frame in the directory, then sample usable detections.
On the device (use `build/tools/` for local build paths):

```sh
./tools/calib_helper --camera_folder=/path/to/frames --detections_output_path=detections.json
./tools/intrinsics_calibrate_disk --detections_path=detections.json --intrinsics_output_path=intrinsics.json
```

The disk calibrator also writes `calibration_coverage.png` (override with
`--coverage_output_path`): selected ChArUco detections in blue on the left and
a smoothed corner-density heatmap on the right, scaled from blue (low) to red
(high) for the selected frames.

The helper processes the directory's frames one at a time and writes all
results, including frames without enough corners, directly to JSON.
The JSON records image dimensions and an array of detections containing
filenames, ChArUco corners and IDs, image points, and object points. The helper
does not run calibration.
`intrinsics_calibrate_disk` randomly selects 50 usable detections without
replacement. Each detection's position is the centroid of its image points
relative to the image center. Positions are stratified into 20 equal radius
bands from the center to an image corner and 20 equal angle sectors, forming
400 blocks. Sampling draws once from each populated block per round, in random
order, until the requested count is reached; exhausted blocks are skipped in
later rounds. If there are too few usable detections, calibration reports an
error. `--max_detections` changes the capture count;
`--max_detections=0` uses all usable results.
The saved JSON can be reused without decoding frames or running detection
again. For live calibration, run
`intrinsics_calibrate --config_path=/path/to/config.json`,
press Enter to capture each frame, and type `q` then Enter to calibrate and quit.
