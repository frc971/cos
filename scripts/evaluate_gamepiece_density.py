#!/usr/bin/env python3
"""Compare relative density and evidence-based ball counts to independent render truth."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--key', default='density')
p.add_argument('--include-hidden', action='store_true', help='Diagnostic only: also count balls with zero rendered pixels')
p.add_argument('--minimum-pixels', type=int, default=1)
a = p.parse_args()

def match(errors, limit):
    n, m = errors.shape
    costs = np.concatenate((np.where(errors <= limit, errors, np.inf), np.full((n, n), n+1.0)), axis=1)
    return [(r, c) for r, c in zip(*linear_sum_assignment(costs)) if c < m]

def area(points):
    cov = np.cov(points.T, bias=True) if len(points) > 1 else np.zeros((2, 2))
    return 2*np.pi*np.sqrt(np.linalg.det(cov + np.eye(2)*.075**2/4))

rows, ratios, geometry_errors, timings, count_errors, support_errors = [], [], [], [], [], []
ranked = {1.5: [], 2.0: [], 3.0: []}
frame_count = hard_hits = hard_predictions = visible_ball_count = 0
hard_radius_hits = 0
for manifest in sorted(a.root.glob('*/manifest.json')):
    truth = json.loads(manifest.read_text())
    balls = np.array([b['translation_m'][:2] for b in truth['gamepieces']])
    ids = np.array([b['cluster_id'] for b in truth['gamepieces']])
    observations = json.loads((manifest.parent/'clusters/clusters.json').read_text())
    for fi, (reference, frame) in enumerate(zip(truth['frames'], observations, strict=True)):
        frame_count += 1
        if 'tracking_time_ms' in frame: timings.append(frame['tracking_time_ms'])
        pose = frame['robot_pose']; angle = pose['yaw']
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        origin = np.array([pose['x'], pose['y']])
        predictions = frame['clusters']
        visible = np.ones(len(balls), bool) if a.include_hidden else np.array(reference['visible_fuel_pixels']) >= a.minimum_pixels
        visible_ball_count += int(sum(visible))
        hard_points = np.array([R @ point + origin for c in predictions for point in c.get('verified_ball_centers_m', [])]).reshape(-1, 2)
        hard_errors = np.linalg.norm(balls[visible, None, :] - hard_points[None, :, :], axis=2)
        hard_hits += len(match(hard_errors, .15))
        hard_radius_hits += len(match(hard_errors, .075))
        hard_predictions += len(hard_points)
        centers, densities, groups, clouds = [], [], [], []
        for group in range(len(truth['fuel_clusters'])):
            cloud = balls[(ids == group) & visible]
            if not len(cloud): continue
            centers.append(cloud.mean(0)); densities.append(len(cloud)/area(cloud))
            groups.append(group); clouds.append(cloud)
        centers = np.array(centers).reshape(-1, 2)
        positions = np.array([R @ c['ground_position_m'] + origin for c in predictions]).reshape(-1, 2)
        errors = np.linalg.norm(centers[:, None, :] - positions[None, :, :], axis=2)
        matched = match(errors, .9)
        if len(matched) != len(centers) or len(predictions) != len(centers):
            geometry_errors.append([manifest.parent.name, fi, len(predictions), len(matched)])
        current = []
        for g, ci in matched:
            c, cloud = predictions[ci], clouds[g]
            estimate = c[a.key]
            if not np.isfinite(estimate) or estimate <= 0:
                raise ValueError(f'Invalid density: {manifest.parent}, frame {fi}, group {groups[g]}')
            row = dict(run=manifest.parent.name, frame=fi, group=groups[g], balls=len(cloud),
                       truth=densities[g], estimated=estimate, silhouette_support=c['visible_fuel_support'])
            support_errors.append(abs(c['visible_fuel_support']-len(cloud))/len(cloud))
            if 'estimated_ball_count' in c:
                row.update(count=c['estimated_ball_count'], verified=c['verified_ball_count'], residual=c['residual_ball_evidence'])
                count_errors.append(abs(np.log(row['count']/len(cloud))))
                row['area'] = c['density_area_m2']
                if 'total_ball_evidence' in c: row['total_evidence'] = c['total_ball_evidence']
            rows.append(row); current.append((densities[g], estimate))
        for (ta, ea), (tb, eb) in itertools.combinations(current, 2):
            ratios.append(abs(np.log((ea/eb)/(ta/tb))))
            for threshold, rankings in ranked.items():
                if max(ta/tb, tb/ta) >= threshold: rankings.append((ta-tb)*(ea-eb) > 0)
if not ratios: p.error('No matched cluster density pairs')
summary = dict(runs=len({r['run'] for r in rows}), frames=frame_count, observations=len(rows), pairs=len(ratios),
               median_ratio_factor=float(np.exp(np.median(ratios))), p90_ratio_factor=float(np.exp(np.percentile(ratios, 90))),
               ranking_accuracy={str(t): float(np.mean(v)) if v else None for t, v in ranked.items()},
               ranked_pairs={str(t): len(v) for t, v in ranked.items()}, geometry_error_frames=len(geometry_errors),
               key=a.key, reference='all balls' if a.include_hidden else f'balls with at least {a.minimum_pixels} rendered pixels')
if count_errors:
    summary.update(count_error_factor_p50_p90=np.exp(np.percentile(count_errors, [50, 90])).tolist(),
                   verified_position_precision=hard_hits/hard_predictions if hard_predictions else None,
                   verified_position_recall=hard_hits/visible_ball_count if visible_ball_count else None,
                   verified_position_precision_7p5cm=hard_radius_hits/hard_predictions if hard_predictions else None,
                   verified_positions=hard_predictions, verified_position_hits=hard_hits)
    summary['absolute_count_error_fraction_p50_p90'] = np.percentile(
        [abs(r['count']-r['balls'])/r['balls'] for r in rows], [50, 90]).tolist()
if timings: summary['tracking_ms_p50_p95'] = np.percentile(timings, [50, 95]).tolist()
summary['silhouette_support_count_error_fraction_p50_p90'] = np.percentile(support_errors, [50, 90]).tolist()
a.output.write_text(json.dumps(dict(summary=summary, geometry_errors=geometry_errors, observations=rows), indent=2)+'\n')
print(json.dumps(summary, indent=2))
