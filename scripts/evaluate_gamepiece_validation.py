#!/usr/bin/env python3
"""Compare tracker output to simulator-only ground truth. Never feeds truth to tracker."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def matches(errors):
    n, m = errors.shape
    costs = np.concatenate((np.where(errors <= .9, errors, np.inf), np.full((n, n), n + 1.0)), axis=1)
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if c < m]

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--expected-clusters', type=int)
p.add_argument('--allow-partial', action='store_true', help='Accept true piles with 1–3 visible balls; still require every pile with 4+')
p.add_argument('--centroid-gate', action='store_true', help='Gate recall on independently rendered centroid visibility; retain all other misses')
a = p.parse_args()
results = []
for manifest in sorted(a.root.glob('*/manifest.json')):
    truth = json.loads(manifest.read_text())
    observations = json.loads((manifest.parent/'clusters/clusters.json').read_text())
    pieces = truth['gamepieces']
    centers = np.array([piece['translation_m'][:2] for piece in pieces]).reshape(-1, 2)
    group_ids = np.array([piece['cluster_id'] for piece in pieces])
    previous_ids, misses, switches, extras = {}, 0, 0, 0
    id_owners, identity_conflicts = {}, 0
    centroid_misses, missed_observations = 0, []
    has_centroids = all('visible_cluster_centroids' in f for f in truth['frames'])
    if a.centroid_gate and not has_centroids:
        p.error('centroid gate requires simulator centroid visibility in every frame')
    distances, shapes, major_ratios, minor_ratios, frames, frame_metrics = [], [], [], [], [], []
    partial_metrics = []
    for frame, observation in zip(truth['frames'], observations, strict=True):
        pose = observation['robot_pose']
        angle = pose['yaw']
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        origin = np.array([pose['x'], pose['y']])
        visibility = np.array(frame['visible_fuel_pixels'])
        targets, covariances, ids = [], [], []
        partial_targets, partial_ids = [], []
        for group in range(len(truth['fuel_clusters'])):
            keep = (group_ids == group) & (visibility >= 8)
            if np.count_nonzero(keep) < 4:
                if np.any(keep):
                    partial_targets.append(centers[keep].mean(0))
                    partial_ids.append(group)
                continue
            # Visible generated centers; no inferred/projection coordinates in truth.
            points = centers[keep]
            targets.append(points.mean(0))
            covariances.append(np.cov(points.T, bias=True))
            ids.append(group)
        predicted = observation['clusters']
        frame_ids = [c['id'] for c in predicted]
        identity_conflicts += len(frame_ids) - len(set(frame_ids))
        points = np.array([rotation @ c['ground_position_m'] + origin for c in predicted]).reshape(-1, 2)
        errors = np.linalg.norm(np.array(targets).reshape(-1, 2)[:, None, :] - points[None, :, :], axis=2)
        matched = 0
        matched_cols = set()
        matched_groups = set()
        for row, col in matches(errors):
            matched += 1
            matched_cols.add(int(col))
            c = predicted[col]
            group = ids[row]
            matched_groups.add(group)
            identity_conflicts += id_owners.get(c['id'], group) != group
            id_owners.setdefault(c['id'], group)
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
        for group in set(ids) - matched_groups:
            visible = bool(frame['visible_cluster_centroids'][group]) if has_centroids else None
            centroid_misses += visible is True
            missed_observations.append([len(frames), int(group), visible])
        partial_matched = 0
        if a.allow_partial and partial_targets:
            remaining = [i for i in range(len(predicted)) if i not in matched_cols]
            partial_errors = np.linalg.norm(np.array(partial_targets)[:, None, :] - points[remaining][None, :, :], axis=2)
            for row, col in matches(partial_errors):
                group, c = partial_ids[row], predicted[remaining[col]]
                identity_conflicts += id_owners.get(c['id'], group) != group
                id_owners.setdefault(c['id'], group)
                if group in previous_ids and previous_ids[group] != c['id']:
                    switches += 1
                previous_ids[group] = c['id']
                partial_metrics.append([len(frames), int(group), c['id'], round(float(partial_errors[row, col]), 4)])
                partial_matched += 1
        extras += len(predicted) - matched - partial_matched
        frames.append([len(targets), len(predicted)])
    def percentile(values, fraction):
        return round(float(np.percentile(values, fraction)), 4) if values else 0
    result = dict(run=manifest.parent.name, frames=len(frames), counts=frames, frame_metrics=frame_metrics,
                  misses=misses, extras=extras, id_switches=switches, identity_conflicts=identity_conflicts,
                  centroid_p95_m=percentile(distances, 95), centroid_max_m=percentile(distances, 100),
                  shape_p95_m=percentile(shapes, 95), shape_max_m=percentile(shapes, 100),
                  minor_std_p05=percentile(minor_ratios, 5), minor_std_p95=percentile(minor_ratios, 95),
                  major_std_p05=percentile(major_ratios, 5), major_std_p95=percentile(major_ratios, 95))
    result['missed_observations'] = missed_observations
    if has_centroids:
        result['centroid_visible_misses'] = centroid_misses
        result['hidden_centroid_misses'] = misses - centroid_misses
    if a.allow_partial:
        result['partial_metrics'] = partial_metrics
        result['partial_centroid_max_m'] = max((m[3] for m in partial_metrics), default=0)
    # Fixed physical acceptance limits. Empty runs must also have no output.
    result['passed'] = (extras == switches == identity_conflicts == 0 and result['centroid_p95_m'] <= .35
                        and result['shape_p95_m'] <= .3 and result['centroid_max_m'] <= .35
                        and result['shape_max_m'] <= .3)
    if a.expected_clusters is not None:
        result['expected_count_met'] = all(n == a.expected_clusters for n, _ in frames)
        result['passed'] &= result['expected_count_met']
    if a.allow_partial:
        result['passed'] &= result['partial_centroid_max_m'] <= .35
    result['strict_passed'] = bool(result['passed'] and misses == 0)
    result['passed'] &= (centroid_misses if a.centroid_gate else misses) == 0
    result['counts'] = [list(pair) for pair in dict.fromkeys(map(tuple, frames))]
    results.append(result)
report = {'passed': bool(results) and all(r['passed'] for r in results),
          'strict_passed': bool(results) and all(r['strict_passed'] for r in results),
          'centroid_gate': a.centroid_gate, 'allow_partial': a.allow_partial, 'runs': results}
print(json.dumps(report, indent=2))
if a.output:
    a.output.write_text(json.dumps(report, separators=(',', ':'))+'\n')
raise SystemExit(0 if report['passed'] else 1)
