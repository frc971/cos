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
            'frames': [{'visible_fuel_pixels': [20]*len(pieces), 'visible_cluster_centroids': [True, True, False]} for _ in range(2)]}
clusters = [{'id': g, 'ground_position_m': pts.mean(0).tolist(),
             'ground_covariance_m2': np.cov(pts.T, bias=True).tolist()} for g, pts in enumerate(groups)]
base = [{'robot_pose': {'x': 0, 'y': 0, 'yaw': 0}, 'clusters': copy.deepcopy(clusters)} for _ in range(2)]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp); run = root/'case'; (run/'clusters').mkdir(parents=True)
    def check(data, expected, partial=True, centroid=False, truth=manifest):
        (run/'manifest.json').write_text(json.dumps(truth))
        (run/'clusters/clusters.json').write_text(json.dumps(data))
        cmd = [sys.executable, str(Path(__file__).with_name('evaluate_gamepiece_validation.py')), str(root)]
        result = subprocess.run(cmd + (['--allow-partial'] if partial else []) +
                                (['--centroid-gate'] if centroid else []), capture_output=True, text=True)
        assert result.returncode in (0, 1), result.stderr
        assert json.loads(result.stdout)['passed'] == expected
    check(base, True); check(base, False, partial=False)
    for defect in ['missing', 'extra', 'position', 'shape', 'id', 'duplicate_id', 'partial_position']:
        data = copy.deepcopy(base)
        if defect == 'missing': data[1]['clusters'].pop(0)
        elif defect == 'extra': data[1]['clusters'].append({'id': 99, 'ground_position_m': [11, 1], 'ground_covariance_m2': [[.01, 0], [0, .01]]})
        elif defect == 'position': data[1]['clusters'][0]['ground_position_m'][0] += .5
        elif defect == 'shape': data[1]['clusters'][0]['ground_covariance_m2'] = [[1, 0], [0, 1]]
        elif defect == 'id': data[1]['clusters'][0]['id'] = 99
        elif defect == 'duplicate_id':
            for frame in data: frame['clusters'][1]['id'] = 0
        else: data[1]['clusters'][2]['ground_position_m'][0] += .5
        check(data, False)
        check(data, False, centroid=True)
    data = copy.deepcopy(base); data[1]['clusters'].pop(0)
    truth = copy.deepcopy(manifest); truth['frames'][1]['visible_cluster_centroids'][0] = False
    check(data, False, truth=truth); check(data, True, centroid=True, truth=truth)
    truth = copy.deepcopy(manifest)
    truth['frames'][0]['visible_fuel_pixels'] = [20]*4 + [0]*7
    truth['frames'][1]['visible_fuel_pixels'] = [0]*4 + [20]*4 + [0]*3
    data = copy.deepcopy(base); data[0]['clusters'] = [copy.deepcopy(clusters[0])]
    data[1]['clusters'] = [copy.deepcopy(clusters[1])]; data[1]['clusters'][0]['id'] = 0
    check(data, False, truth=truth); check(data, False, centroid=True, truth=truth)
print('Evaluator controls passed: strict/centroid visibility, eight defects, and partial detections.')
