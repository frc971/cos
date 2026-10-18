#!/usr/bin/env python3
"""Run on dev-orin from /root after deploying. Uses only simulated images."""
import argparse
import json
import math
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/root/cluster-validation/suite'))
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--seeds', default='7,23')
args = parser.parse_args()
scenarios = json.loads(Path('/root/cluster-validation/scenarios.json').read_text())
args.root.mkdir(parents=True, exist_ok=True)
poses = [{'translation_m': [5.3 + .5*i/12, 4 + .1*math.sin(i/12*math.pi*2)],
          'rotation_rad': -.12 + .24*i/12} for i in range(13)]
(args.root/'poses.json').write_text(json.dumps({'poses': poses}))
for name, clusters in scenarios.items():
    for seed in map(int, args.seeds.split(',')):
        run = args.root / f'{name}-{seed}'
        command = ['./tools/path_camera_sim', f'--poses={args.root}/poses.json',
                   '--camera_config=constants/gamepiece/camera.json', '--fuel_truth',
                   '--fuel_clusters='+json.dumps(clusters), f'--fuel_seed={seed}',
                   f'--fuel_noise={args.noise}', f'--output_dir={run}']
        with (args.root/f'{name}-{seed}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(['./tools/hsv_cluster_visualizer', f'--sim_run={run}',
                            f'--output_dir={run}/clusters'], check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        print(run, flush=True)
