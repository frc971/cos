#!/usr/bin/env python3
"""Negative controls for the acceptance evaluator, independent of tracker output."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

groups = [np.array([[6, 2], [6.2, 2], [6, 2.2], [6.2, 2.2]]),
          np.array([[8, 4], [8.2, 4], [8, 4.2], [8.2, 4.2]]),
          np.array([[10, 6], [10.2, 6], [10, 6.2]])]
pieces = [{'cluster_id': g, 'translation_m': [*p, .075]} for g, pts in enumerate(groups) for p in pts]
manifest = {'gamepieces': pieces, 'fuel_clusters': [{}, {}, {}],
            'frames': [{'visible_fuel_pixels': [20]*len(pieces)} for _ in range(2)]}
clusters = [{'id': g, 'ground_position_m': pts.mean(0).tolist(),
             'ground_covariance_m2': np.cov(pts.T, bias=True).tolist()} for g, pts in enumerate(groups)]
base = [{'robot_pose': {'x': 0, 'y': 0, 'yaw': 0}, 'clusters': copy.deepcopy(clusters)} for _ in range(2)]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp); run = root/'case'; (run/'clusters').mkdir(parents=True)
    (run/'manifest.json').write_text(json.dumps(manifest))
    def check(data, expected, partial=True):
        (run/'clusters/clusters.json').write_text(json.dumps(data))
        cmd = [sys.executable, str(Path(__file__).with_name('evaluate_gamepiece_validation.py')), str(root)]
        result = subprocess.run(cmd + (['--allow-partial'] if partial else []), capture_output=True, text=True)
        assert result.returncode in (0, 1), result.stderr
        assert json.loads(result.stdout)['passed'] == expected
    check(base, True); check(base, False, partial=False)
    for defect in ['missing', 'extra', 'position', 'shape', 'id', 'partial_position']:
        data = copy.deepcopy(base)
        if defect == 'missing': data[1]['clusters'].pop(0)
        elif defect == 'extra': data[1]['clusters'].append({'id': 99, 'ground_position_m': [11, 1], 'ground_covariance_m2': [[.01, 0], [0, .01]]})
        elif defect == 'position': data[1]['clusters'][0]['ground_position_m'][0] += .5
        elif defect == 'shape': data[1]['clusters'][0]['ground_covariance_m2'] = [[1, 0], [0, 1]]
        elif defect == 'id': data[1]['clusters'][0]['id'] = 99
        else: data[1]['clusters'][2]['ground_position_m'][0] += .5
        check(data, False)
print('Evaluator positive/strict controls and six defect controls passed.')
