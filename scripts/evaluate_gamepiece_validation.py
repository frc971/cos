#!/usr/bin/env python3
"""Compare tracker output to simulator-only ground truth. Never feeds truth to tracker."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--expected-clusters', type=int)
a = p.parse_args()
results = []
for manifest in sorted(a.root.glob('*/manifest.json')):
    truth = json.loads(manifest.read_text())
    observations = json.loads((manifest.parent/'clusters/clusters.json').read_text())
    pieces = truth['gamepieces']
    centers = np.array([piece['translation_m'][:2] for piece in pieces]).reshape(-1, 2)
    group_ids = np.array([piece['cluster_id'] for piece in pieces])
    previous_ids, misses, switches, extras = {}, 0, 0, 0
    distances, shapes, major_ratios, minor_ratios, frames, frame_metrics = [], [], [], [], [], []
    for frame, observation in zip(truth['frames'], observations, strict=True):
        pose = observation['robot_pose']
        angle = pose['yaw']
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        origin = np.array([pose['x'], pose['y']])
        visibility = np.array(frame['visible_fuel_pixels'])
        targets, covariances, ids = [], [], []
        for group in range(len(truth['fuel_clusters'])):
            keep = (group_ids == group) & (visibility >= 8)
            if np.count_nonzero(keep) < 4:
                continue
            # Visible generated centers; no inferred/projection coordinates in truth.
            points = centers[keep]
            targets.append(points.mean(0))
            covariances.append(np.cov(points.T, bias=True))
            ids.append(group)
        predicted = observation['clusters']
        points = np.array([rotation @ c['ground_position_m'] + origin for c in predicted]).reshape(-1, 2)
        errors = np.linalg.norm(np.array(targets).reshape(-1, 2)[:, None, :] - points[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(errors)
        matched = 0
        for row, col in zip(rows, cols):
            if errors[row, col] > .9:
                continue
            matched += 1
            c = predicted[col]
            group = ids[row]
            if group in previous_ids and previous_ids[group] != c['id']:
                switches += 1
            previous_ids[group] = c['id']
            distances.append(float(errors[row, col]))
            covariance = rotation @ np.array(c['ground_covariance_m2']) @ rotation.T
            expected = covariances[row]
            vals, vectors = np.linalg.eigh(expected)
            sqrt = (vectors * np.sqrt(np.maximum(vals, 0))) @ vectors.T
            cross = np.linalg.eigvalsh(sqrt @ covariance @ sqrt)
            shapes.append(float(np.sqrt(max(0, np.trace(covariance + expected) - 2*np.sqrt(np.maximum(cross, 0)).sum()))))
            frame_metrics.append([len(frames), int(group), c["id"], round(distances[-1], 4), round(shapes[-1], 4)])
            ratios = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0) / np.maximum(vals, .0025))
            minor_ratios.append(float(ratios[0])); major_ratios.append(float(ratios[1]))
        misses += len(targets) - matched
        extras += len(predicted) - matched
        frames.append([len(targets), len(predicted)])
    def percentile(values, fraction):
        return round(float(np.percentile(values, fraction)), 4) if values else 0
    result = dict(run=manifest.parent.name, frames=len(frames), counts=frames, frame_metrics=frame_metrics,
                  misses=misses, extras=extras, id_switches=switches,
                  centroid_p95_m=percentile(distances, 95), centroid_max_m=percentile(distances, 100),
                  shape_p95_m=percentile(shapes, 95), shape_max_m=percentile(shapes, 100),
                  minor_std_p05=percentile(minor_ratios, 5), minor_std_p95=percentile(minor_ratios, 95),
                  major_std_p05=percentile(major_ratios, 5), major_std_p95=percentile(major_ratios, 95))
    # Fixed physical acceptance limits. Empty runs must also have no output.
    result['passed'] = (misses == extras == switches == 0 and result['centroid_p95_m'] <= .35
                        and result['shape_p95_m'] <= .3 and result['centroid_max_m'] <= .35
                        and result['shape_max_m'] <= .3)
    if a.expected_clusters is not None:
        result['expected_count_met'] = all(n == a.expected_clusters for n, _ in frames)
        result['passed'] &= result['expected_count_met']
    result['counts'] = [list(pair) for pair in dict.fromkeys(map(tuple, frames))]
    results.append(result)
report = {'passed': bool(results) and all(r['passed'] for r in results), 'runs': results}
print(json.dumps(report, indent=2))
if a.output:
    a.output.write_text(json.dumps(report, separators=(',', ':'))+'\n')
raise SystemExit(0 if report['passed'] else 1)
