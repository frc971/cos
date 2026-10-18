#!/usr/bin/env python3
"""Replay unchanged simulator RGB through the deployed C++ tracker, from /root."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--visualizer', default='./tools/hsv_cluster_visualizer')
a = p.parse_args()
manifests = sorted(a.source.glob('*/manifest.json'))
if not manifests:
    p.error('source contains no simulated runs')
for manifest in manifests:
    target = a.output / manifest.parent.name
    target.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest.read_text())
    for frame in data['frames']:
        frame['file'] = str((manifest.parent / frame['file']).resolve())
    (target/'manifest.json').write_text(json.dumps(data))
    with (target/'replay.log').open('w') as log:
        subprocess.run([a.visualizer, f'--sim_run={target}',
                        f'--output_dir={target}/clusters'], check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    print(target, flush=True)
